from transformers.models.mixtral.modeling_mixtral import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path
import json
from tqdm import tqdm
import os
from accelerate import init_empty_weights
from functools import partial

def get_free_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return 'cpu'
    free_memory = [torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)]
    most_free_gpu_index = int(torch.argmax(torch.tensor(free_memory)))
    return f'cuda:{most_free_gpu_index}'

def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params



def get_expert_frequency(model, tokenizer, model_name, dataset_name, split, seed, max_samples = None, batch_size = 32):
    selected_layers = list(range(len(model.model.layers)))

    device = next(model.parameters()).device
    expert_selection_counts = {i: torch.zeros(model.model.config.num_local_experts, device=device) for i in selected_layers}

    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        text_column = 'text'
    elif 'ptb' in dataset_name:
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split=split)
        text_column = 'sentence'
    elif 'c4' in dataset_name:
        dataset = load_dataset("json", data_files="/aifs4su/lilujun/SVD-MoE-merge/SVD-MOE-new/function_base/c4-train.json")[split]
        text_column = 'text'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    handles = []

    def hook_for_expert_counting(module, input, output, module_name=None):
        if isinstance(module, MixtralSparseMoeBlock):
            router_logits = output[1]  # Assuming the router logits are the second output
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1)

            for expert_idx in selected_experts.unique():
                if expert_idx.item() < expert_selection_counts[module_name].size(0):
                    expert_selection_counts[module_name][expert_idx.item()] += (selected_experts == expert_idx).sum().item()
                else:
                    logger.warning(f"Expert index {expert_idx.item()} out of range for module {module_name}")

    def create_hook(layer_idx):
        """
        Creates a partial hook function for a specific layer.
        """
        return partial(hook_for_expert_counting, module_name=layer_idx)

    # Register hooks for each expert in each selected layer
    for layer_idx in selected_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'block_sparse_moe'):
            moe_module = layer.block_sparse_moe
            handle = moe_module.register_forward_hook(create_hook(layer_idx))
            handles.append(handle)
        # elif hasattr(layer, 'gate'):
        #     gate_module = layer.gate
        #     handle = gate_module.register_forward_hook(create_hook(layer_idx))
        #     handles.append(handle)

    # Iterate through the dataloader and perform forward passes to collect counts
    for batch in tqdm(dataloader, desc="Collecting expert activation counts"):
        inputs = tokenizer(batch[text_column], truncation=True, padding=True, max_length=2048, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    # Remove all hooks after collection
    for handle in handles:
        handle.remove()

    # Save the counts to a JSON file
    counts_dict = {layer: counts.tolist() for layer, counts in expert_selection_counts.items()}

    with open(f"/aifs4su/lilujun/SVD-MoE-merge/MoE/{model_name}_{dataset_name}_{max_samples}_expert_frequencies.json", "w") as f:
        json.dump(counts_dict, f, indent=4)















def save_model(model, path):
    """Saves the model to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")



def load_model_tqdm(checkpoint_path, base_model_path, ratio=0.35):
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(base_model_path, 
                                                    device_map="auto", 
                                                    trust_remote_code=True, 
                                                    torch_dtype=torch.bfloat16)
        
    
    for i in tqdm(range(len(model.model.layers)), desc="Initializing layers"):
        model.model.layers[i].block_sparse_moe = Merge_MixtralSparseMoeBlock(
            model.model.layers[i].block_sparse_moe.config, 
            ratio=ratio, 
            expert_freq=None
        ).to(model.model.layers[i].block_sparse_moe.gate.weight.device)
    
    # print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    
    # print("Loading state dict...")
    # 创建一个进度条来显示加载进度
    pbar = tqdm(total=len(checkpoint), desc="Loading checkpoint")
    for k, v in checkpoint.items():
        model.state_dict()[k].copy_(v)
        pbar.update(1)
    pbar.close()
    
    return model

import random
import numpy as np
seed = 42  # or any other integer
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class Merge_MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config, share_ratio, delta_ratio, expert_freq):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # added
        self.share_ratio = share_ratio
        self.delta_ratio = delta_ratio
        self.expert_mean = {"w1_mean": None, "w2_mean": None, "w3_mean": None}
        self.expert_freq = expert_freq
        self.config = config

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, dtype=torch.bfloat16)
        
        # self.experts = nn.ModuleList([delta_MixtralBlockSparseTop2MLP(config, ratio) for _ in range(self.num_experts)])
        # self.shared_experts = delta_MixtralBlockSparseTop2MLP(config, ratio) 
        # self.shared_experts = MixtralBlockSparseTop2MLP(config)
        # self.shared_experts = share_expert_with_delta_weight(config, ratio) 

        self.experts = nn.ModuleList([share_expert_with_delta_weight(config, share_ratio, delta_ratio)  for _ in range(self.num_experts)])
        # self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config)  for _ in range(self.num_experts)])

        # Jitter parametersmodel2.model.layers[i].block_sparse_moe.num_experts
        self.jitter_noise = config.router_jitter_noise

        self.expert_frequency = [0] * self.num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() > 0:
                self.expert_frequency[expert_idx] += 1
                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def get_expert_frequencies(self):
        """返回每个专家的激活频率"""
        return self.expert_frequency

    def reset_expert_frequencies(self):
        """重置每个专家的激活频率为零"""
        self.expert_frequency = [0] * self.num_experts

    @staticmethod
    @torch.no_grad()
    def svd_delta(W, ratio=1):
        U, S, VT = torch.linalg.svd(W.float(), full_matrices=False)
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        del W
        truc_s = S[:num_s_after_trunc]
        del S
        truc_u = U[:, :num_s_after_trunc]
        del U
        truc_v = VT[:num_s_after_trunc, :]
        del VT
        truc_sigma = torch.diag(truc_s)
        del truc_s
        #### Replace Attn, MLP ####
        sqrtSigma = torch.sqrt(truc_sigma)
        # sqrtSigma = truc_sigma
        svd_u = torch.matmul(truc_u, sqrtSigma)
        svd_v = torch.matmul(sqrtSigma, truc_v)

        return svd_u.to(torch.bfloat16), svd_v.to(torch.bfloat16)

    # @staticmethod
    # @torch.no_grad()
    # def svd_delta(W, ratio=1):
    #     U, S, VT = torch.linalg.svd(W.float(), full_matrices=False)
    #     num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
    #     del W
    #     truc_s = S[:num_s_after_trunc]
    #     del S
    #     truc_u = U[:, :num_s_after_trunc]
    #     del U
    #     truc_v = VT[:num_s_after_trunc, :]
    #     del VT
    #     truc_sigma = torch.diag(truc_s)
    #     del truc_s
    #     result = truc_u @ truc_sigma @ truc_v
    #     return result.to(torch.bfloat16)


    @torch.no_grad()
    def merge_experts(self, module):
        self.gate.weight.data = module.gate.weight.data

        total_weight = 0 
        for j in range(self.num_experts):
            w1_weight = module.experts[j].w1.weight
            w2_weight = module.experts[j].w2.weight
            w3_weight = module.experts[j].w3.weight
            freq = self.expert_freq[j]

            if self.expert_mean["w1_mean"] is None:
                self.expert_mean["w1_mean"] = w1_weight * freq
            else:
                self.expert_mean["w1_mean"] += w1_weight * freq

            if self.expert_mean["w2_mean"] is None:
                self.expert_mean["w2_mean"] = w2_weight * freq
            else:
                self.expert_mean["w2_mean"] += w2_weight * freq

            if self.expert_mean["w3_mean"] is None:
                self.expert_mean["w3_mean"] = w3_weight * freq  
            else:
                self.expert_mean["w3_mean"] += w3_weight * freq 
            total_weight += freq

        self.expert_mean["w1_mean"] /= total_weight
        self.expert_mean["w2_mean"] /= total_weight
        self.expert_mean["w3_mean"] /= total_weight
        del total_weight

        shared_w1 = nn.Parameter(self.expert_mean["w1_mean"])
        shared_w2 = nn.Parameter(self.expert_mean["w2_mean"])
        shared_w3 = nn.Parameter(self.expert_mean["w3_mean"])



        # self.expert_mean["w1_u1"], self.expert_mean["w1_v1"] = self.svd_delta(self.expert_mean["w1_mean"], ratio=self.share_ratio)
        # self.expert_mean["w2_u2"], self.expert_mean["w2_v2"] = self.svd_delta(self.expert_mean["w2_mean"], ratio=self.share_ratio)
        # self.expert_mean["w3_u3"], self.expert_mean["w3_v3"] = self.svd_delta(self.expert_mean["w3_mean"], ratio=self.share_ratio)

        # shared_w1_u1 = nn.Parameter(self.expert_mean["w1_u1"])
        # shared_w1_v1 = nn.Parameter(self.expert_mean["w1_v1"])
        # shared_w2_u2 = nn.Parameter(self.expert_mean["w2_u2"])
        # shared_w2_v2 = nn.Parameter(self.expert_mean["w2_v2"])
        # shared_w3_u3 = nn.Parameter(self.expert_mean["w3_u3"])
        # shared_w3_v3 = nn.Parameter(self.expert_mean["w3_v3"])



        for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):
            # w1_weight = module.experts[j].w1.weight
            # w2_weight = module.experts[j].w2.weight
            # w3_weight = module.experts[j].w3.weight
            # self.experts[j].w1.weight.data = self.expert_mean["w1_mean"] + self.svd_delta(w1_weight - self.expert_mean["w1_mean"], ratio=self.ratio)
            # self.experts[j].w2.weight.data = self.expert_mean["w2_mean"] + self.svd_delta(w2_weight - self.expert_mean["w2_mean"], ratio=self.ratio)
            # self.experts[j].w3.weight.data = self.expert_mean["w3_mean"] + self.svd_delta(w3_weight - self.expert_mean["w3_mean"], ratio=self.ratio)
            
            # self.experts[j].w1.weight = self.expert_mean["w1_mean"]
            # self.experts[j].w2.weight = self.expert_mean["w2_mean"]
            # self.experts[j].w3.weight = self.expert_mean["w3_mean"]

            self.experts[j].w1.weight = shared_w1
            self.experts[j].w2.weight = shared_w2
            self.experts[j].w3.weight = shared_w3

            # self.experts[j].w1_u1.weight = shared_w1_u1
            # self.experts[j].w1_v1.weight = shared_w1_v1
            # self.experts[j].w2_u2.weight = shared_w2_u2
            # self.experts[j].w2_v2.weight = shared_w2_v2
            # self.experts[j].w3_u3.weight = shared_w3_u3
            # self.experts[j].w3_v3.weight = shared_w3_v3
            
            # self.experts[j].delta_w1.weight.data = self.svd_delta(w1_weight - self.expert_mean["w1_mean"], ratio=self.ratio)
            # self.experts[j].delta_w2.weight.data = self.svd_delta(w2_weight - self.expert_mean["w2_mean"], ratio=self.ratio)
            # self.experts[j].delta_w3.weight.data = self.svd_delta(w3_weight - self.expert_mean["w3_mean"], ratio=self.ratio)
            # self.experts[j].u1_weight.data, self.experts[j].v1_weight.data = self.svd_delta(module.experts[j].w1.weight - self.expert_mean["w1_mean"], ratio=self.ratio)
            # self.experts[j].u2_weight.data, self.experts[j].v2_weight.data = self.svd_delta(module.experts[j].w2.weight - self.expert_mean["w2_mean"], ratio=self.ratio)
            # self.experts[j].u3_weight.data, self.experts[j].v3_weight.data = self.svd_delta(module.experts[j].w3.weight - self.expert_mean["w3_mean"], ratio=self.ratio)
            self.experts[j].u1.weight.data, self.experts[j].v1.weight.data = self.svd_delta(module.experts[j].w1.weight - self.expert_mean["w1_mean"], ratio=self.delta_ratio)
            self.experts[j].u2.weight.data, self.experts[j].v2.weight.data = self.svd_delta(module.experts[j].w2.weight - self.expert_mean["w2_mean"], ratio=self.delta_ratio)
            self.experts[j].u3.weight.data, self.experts[j].v3.weight.data = self.svd_delta(module.experts[j].w3.weight - self.expert_mean["w3_mean"], ratio=self.delta_ratio)


# class delta_weight():
#     def __init__(self, config: MixtralConfig, ratio=1):
#         super().__init__()
#         self.intermediate_dim = config.intermediate_size
#         self.hidden_dim = config.hidden_size

#         self.ratio = ratio
#         self.low_rank = int(self.intermediate_dim * self.hidden_dim * self.ratio / (self.intermediate_dim + self.hidden_dim))

#         self.u1_weight = nn.Parameter(torch.empty(self.low_rank, self.intermediate_dim, dtype=torch.bfloat16))
#         self.v1_weight = nn.Parameter(torch.empty(self.hidden_dim, self.low_rank, dtype=torch.bfloat16))

#         self.u2_weight = nn.Parameter(torch.empty(self.low_rank, self.hidden_dim, dtype=torch.bfloat16))
#         self.v2_weight = nn.Parameter(torch.empty(self.intermediate_dim, self.low_rank, dtype=torch.bfloat16))

#         self.u3_weight = nn.Parameter(torch.empty(self.low_rank, self.intermediate_dim, dtype=torch.bfloat16))
#         self.v3_weight = nn.Parameter(torch.empty(self.hidden_dim, self.low_rank, dtype=torch.bfloat16))

# class delta_MixtralBlockSparseTop2MLP(MixtralBlockSparseTop2MLP, delta_weight):
#     def __init__(self, config: MixtralConfig, ratio=1):
#         MixtralBlockSparseTop2MLP.__init__(self, config)
#         delta_weight.__init__(self, config, ratio)

#     def forward(self, hidden_states):
#         up = self.w3(hidden_states) + (hidden_states @ self.v3_weight.t()) @ self.u3_weight.t()
        

#         gate = self.w1(hidden_states) + (hidden_states @ self.v1_weight.t()) @ self.u1_weight.t()

#         return self.w2(self.act_fn(gate) * up) + (self.act_fn(gate) * up @ self.v2_weight.t()) @ self.u2_weight.t()


class delta_weight_linear(nn.Module):
    def __init__(self, config: MixtralConfig, ratio=1):
        super().__init__()
        self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = torch.bfloat16

        self.w1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)
        self.w2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=self.dtype)
        self.w3 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)

        self.act_fn = ACT2FN[config.hidden_act]

        self.delta_w1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.delta_w2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.delta_w3 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states):
        up = self.w3(hidden_states) + self.delta_w3(hidden_states)
        gate = self.w1(hidden_states) + self.delta_w1(hidden_states)
        return self.w2(self.act_fn(gate) * up) + self.delta_w2(self.act_fn(gate) * up)


class share_expert_with_delta_weight(nn.Module):
    def __init__(self, config: MixtralConfig, share_ratio=1, delta_ratio=1):
        super().__init__()
        self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = torch.bfloat16

        self.w1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)
        self.w2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=self.dtype)
        self.w3 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)

        self.act_fn = ACT2FN[config.hidden_act]

        self.delta_ratio = delta_ratio
        self.delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))

        self.u1 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v1 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        self.u2 = nn.Linear(self.delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v2 = nn.Linear(self.intermediate_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        self.u3 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v3 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        # v is right, u is left

    def forward(self, hidden_states):
        up = self.w3(hidden_states) + self.u3(self.v3(hidden_states))
        gate = self.w1(hidden_states) + self.u1(self.v1(hidden_states))
        return self.w2(self.act_fn(gate) * up) + self.u2(self.v2(self.act_fn(gate) * up))
    

class share_svd_expert_with_delta_weight(nn.Module):
    def __init__(self, config: MixtralConfig, share_ratio=1, delta_ratio=1):
        super().__init__()
        self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = torch.bfloat16
        self.share_ratio = share_ratio
        self.share_low_rank = int(self.intermediate_dim * self.hidden_dim * self.share_ratio / (self.intermediate_dim + self.hidden_dim))


        self.w1_u1 = nn.Linear(self.share_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.w1_v1 = nn.Linear(self.hidden_dim, self.share_low_rank, bias=False, dtype=torch.bfloat16)

        self.w2_u2 = nn.Linear(self.share_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.w2_v2 = nn.Linear(self.intermediate_dim, self.share_low_rank, bias=False, dtype=torch.bfloat16)

        self.w3_u3 = nn.Linear(self.share_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.w3_v3 = nn.Linear(self.hidden_dim, self.share_low_rank, bias=False, dtype=torch.bfloat16)


        self.act_fn = ACT2FN[config.hidden_act]

        self.delta_ratio = delta_ratio
        self.delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))

        self.u1 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v1 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        self.u2 = nn.Linear(self.delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v2 = nn.Linear(self.intermediate_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        self.u3 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v3 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        # v is right, u is left

    def forward(self, hidden_states):
        up = self.w3_u3(self.w3_v3(hidden_states)) + self.u3(self.v3(hidden_states))
        gate = self.w1_u1(self.w1_v1(hidden_states)) + self.u1(self.v1(hidden_states))
        return self.w2_u2(self.w2_v2(self.act_fn(gate) * up)) + self.u2(self.v2(self.act_fn(gate) * up))
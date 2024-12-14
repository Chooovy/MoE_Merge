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


class Merge_MixtralSparseMoeBlock_debug(nn.Module):
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

    def __init__(self, config, ratio, expert_freq):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # added
        self.ratio = ratio
        self.expert_mean = {"w1_mean": None, "w2_mean": None, "w3_mean": None}
        self.expert_freq = expert_freq
        self.config = config

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, dtype=torch.bfloat16)
        
        # self.experts = nn.ModuleList([delta_MixtralBlockSparseTop2MLP(config, ratio) for _ in range(self.num_experts)])
        # self.shared_experts = delta_MixtralBlockSparseTop2MLP(config, ratio) 
        # self.shared_experts = MixtralBlockSparseTop2MLP(config)
        # self.shared_experts = share_expert_with_delta_weight(config, ratio) 

        self.experts = nn.ModuleList([share_expert_with_delta_weight(config, ratio)  for _ in range(self.num_experts)])
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

        # self.shared_experts.w1.weight.data = self.expert_mean["w1_mean"]
        # self.shared_experts.w2.weight.data = self.expert_mean["w2_mean"]
        # self.shared_experts.w3.weight.data = self.expert_mean["w3_mean"]

        for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):
            w1_weight = module.experts[j].w1.weight
            w2_weight = module.experts[j].w2.weight
            w3_weight = module.experts[j].w3.weight
            # self.experts[j].w1.weight.data = self.expert_mean["w1_mean"] + self.svd_delta(w1_weight - self.expert_mean["w1_mean"], ratio=self.ratio)
            # self.experts[j].w2.weight.data = self.expert_mean["w2_mean"] + self.svd_delta(w2_weight - self.expert_mean["w2_mean"], ratio=self.ratio)
            # self.experts[j].w3.weight.data = self.expert_mean["w3_mean"] + self.svd_delta(w3_weight - self.expert_mean["w3_mean"], ratio=self.ratio)
            self.experts[j].w1.weight.data = self.expert_mean["w1_mean"]
            self.experts[j].w2.weight.data = self.expert_mean["w2_mean"]
            self.experts[j].w3.weight.data = self.expert_mean["w3_mean"]
            
            # self.experts[j].delta_w1.weight.data = self.svd_delta(w1_weight - self.expert_mean["w1_mean"], ratio=self.ratio)
            # self.experts[j].delta_w2.weight.data = self.svd_delta(w2_weight - self.expert_mean["w2_mean"], ratio=self.ratio)
            # self.experts[j].delta_w3.weight.data = self.svd_delta(w3_weight - self.expert_mean["w3_mean"], ratio=self.ratio)
            # self.experts[j].u1_weight.data, self.experts[j].v1_weight.data = self.svd_delta(module.experts[j].w1.weight - self.expert_mean["w1_mean"], ratio=self.ratio)
            # self.experts[j].u2_weight.data, self.experts[j].v2_weight.data = self.svd_delta(module.experts[j].w2.weight - self.expert_mean["w2_mean"], ratio=self.ratio)
            # self.experts[j].u3_weight.data, self.experts[j].v3_weight.data = self.svd_delta(module.experts[j].w3.weight - self.expert_mean["w3_mean"], ratio=self.ratio)
            self.experts[j].u1.weight.data, self.experts[j].v1.weight.data = self.svd_delta(module.experts[j].w1.weight - self.expert_mean["w1_mean"], ratio=self.ratio)
            self.experts[j].u2.weight.data, self.experts[j].v2.weight.data = self.svd_delta(module.experts[j].w2.weight - self.expert_mean["w2_mean"], ratio=self.ratio)
            self.experts[j].u3.weight.data, self.experts[j].v3.weight.data = self.svd_delta(module.experts[j].w3.weight - self.expert_mean["w3_mean"], ratio=self.ratio)


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
    def __init__(self, config: MixtralConfig, ratio=1):
        super().__init__()
        self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = torch.bfloat16

        self.w1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)
        self.w2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=self.dtype)
        self.w3 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)

        self.act_fn = ACT2FN[config.hidden_act]

        self.ratio = ratio
        self.low_rank = int(self.intermediate_dim * self.hidden_dim * self.ratio / (self.intermediate_dim + self.hidden_dim))

        self.u1 = nn.Linear(self.low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v1 = nn.Linear(self.hidden_dim, self.low_rank, bias=False, dtype=torch.bfloat16)

        self.u2 = nn.Linear(self.low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v2 = nn.Linear(self.intermediate_dim, self.low_rank, bias=False, dtype=torch.bfloat16)

        self.u3 = nn.Linear(self.low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v3 = nn.Linear(self.hidden_dim, self.low_rank, bias=False, dtype=torch.bfloat16)

        # v is right, u is left

    def forward(self, hidden_states):
        up = self.w3(hidden_states) + self.u3(self.v3(hidden_states))
        gate = self.w1(hidden_states) + self.u1(self.v1(hidden_states))
        return self.w2(self.act_fn(gate) * up) + self.u2(self.v2(self.act_fn(gate) * up))
    # def forward(self, hidden_states):
    #     up = self.w3(hidden_states)
    #     gate = self.w1(hidden_states)
    #     return self.w2(self.act_fn(gate) * up)
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
import random
from .model_utils import find_layers, find_linear_layers


def cal_scale_inv(svd_scale):
    try:
        scale_inv = torch.linalg.inv(svd_scale)
    except Exception as e:
        print("Warning: svd_scale is not full rank!")
        svd_scale += 1e-6 * torch.eye(svd_scale.shape[0]).to(svd_scale.device)
        scale_inv = torch.linalg.inv(svd_scale)
    return scale_inv.float()

def get_free_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return 'cpu'
    free_memory = [torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)]
    most_free_gpu_index = int(torch.argmax(torch.tensor(free_memory)))
    return f'cuda:{most_free_gpu_index}'

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




@torch.no_grad()
def get_svd_scale(model, tokenizer, model_name, dataset_name = 'wikitext', split = 'train', seed = 42, seqlen = 2048, batch_size = 1, max_samples = None):

    layers = model.model.layers
    layers[0] = layers[0]

    cache_file = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/calib_loader_{model_name}_{dataset_name}_{max_samples}.pt"

    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        text_column = 'text'
        tot_text = "\n\n".join(dataset[text_column])
    elif 'ptb' in dataset_name:
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split=split)
        text_column = 'sentence'
        tot_text = "\n\n".join(dataset[text_column])
    elif 'c4' in dataset_name:
        dataset = load_dataset("json", data_files="/aifs4su/lilujun/SVD-MoE-merge/SVD-MOE-new/function_base/c4-train.json")[split]
        text_column = 'text'
        tot_text = "\n\n".join(dataset[text_column])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    calib_loader = []
    for s in range(max_samples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                calib_loader.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)

    torch.save(calib_loader, cache_file)

    # calib_loader = torch.load("/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/calib_loader_SmolLlamix-8x101M_wikitext_1000.pt")



    dtype = next(iter(model.parameters())).dtype
    device = get_free_gpu()

    inps = torch.zeros(
        (len(calib_loader), seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.detach()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].detach()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].detach()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].detach()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].detach()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])

    for batch in calib_loader:
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    
    layers[0] = layers[0].module


    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids'].detach()
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer = layers[i]
        # subset = find_layers(module = layer, layers=[nn.Linear, MixtralSparseMoeBlock], process_moe_block=True) 
        subset = find_linear_layers(module = layer, layers=[nn.Linear])

        process_subset = {}
        for name, module in subset.items():
            if 'experts' in name:
                process_subset[name] = module

        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        
        handles = []
        
        for name in process_subset:
            process_subset[name].scaling_diag_matrix = 0
            handles.append(process_subset[name].register_forward_hook(hook))

        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(device))[0]
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()
        for name in process_subset:
            device = get_free_gpu()
            raw_scaling_diag_matrix = process_subset[name].scaling_diag_matrix.double().to(device)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(device)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix
            scaling_diag_matrix = raw_scaling_diag_matrix = process_subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, process_subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat











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

        self.experts = nn.ModuleList([share_svd_expert_with_delta_weight(config, share_ratio, delta_ratio)  for _ in range(self.num_experts)])
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
    def svd_delta(W, ratio=1, svd_scale=None, rank=None):
        if rank is None:
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        else:
            num_s_after_trunc = rank
        if svd_scale is None:
            U, S, VT = torch.linalg.svd(W.float(), full_matrices=False)
            del W
            truc_s = S[:num_s_after_trunc]
            del S
            truc_u = U[:, :num_s_after_trunc]
            del U
            truc_v = VT[:num_s_after_trunc, :]
            del VT
            truc_sigma = torch.diag(truc_s)
            del truc_s
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma)
            svd_v = torch.matmul(sqrtSigma, truc_v)
        else:
            W_scale = torch.matmul(W, svd_scale.bfloat16())
            U, S, VT = torch.linalg.svd(W_scale.float(), full_matrices=False)
            del W_scale
            truc_s = S[:num_s_after_trunc]
            del S
            truc_u = U[:, :num_s_after_trunc]
            del U
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], cal_scale_inv(svd_scale))
            del VT
            truc_sigma = torch.diag(truc_s)
            del truc_s
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma)
            svd_v = torch.matmul(sqrtSigma, truc_v)
            
        return svd_u.to(torch.bfloat16), svd_v.to(torch.bfloat16)

    @staticmethod
    @torch.no_grad()
    def svd_delta_reslut(W, ratio=1, svd_scale=None):
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        if svd_scale is None:
            U, S, VT = torch.linalg.svd(W.float(), full_matrices=False)
            del W
            truc_s = S[:num_s_after_trunc]
            del S
            truc_u = U[:, :num_s_after_trunc]
            del U
            truc_v = VT[:num_s_after_trunc, :]
            del VT
            truc_sigma = torch.diag(truc_s)
            del truc_s
            result = truc_u @ truc_sigma @ truc_v
        else:
            W_scale = torch.matmul(W, svd_scale.bfloat16())
            U, S, VT = torch.linalg.svd(W_scale.float(), full_matrices=False)
            del W_scale
            truc_s = S[:num_s_after_trunc]
            del S
            truc_u = U[:, :num_s_after_trunc]
            del U
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], cal_scale_inv(svd_scale))
            del VT
            truc_sigma = torch.diag(truc_s)
            del truc_s
            result = truc_u @ truc_sigma @ truc_v
        return result.to(torch.bfloat16)


    @torch.no_grad()
    def merge_experts(self, module, svd_scale = None):
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

        scale_w1_mean = None
        scale_w2_mean = None
        scale_w3_mean = None
        total_freq = 0
        if svd_scale is not None:
            for j in range(self.num_experts):
                base_name = f"block_sparse_moe.experts.{j}."
                freq = self.expert_freq[j]  
                total_freq += freq
                if scale_w1_mean is None:
                    scale_w1_mean = svd_scale[base_name + "w1"] * freq
                else:
                    scale_w1_mean += svd_scale[base_name + "w1"] * freq
                if scale_w2_mean is None:
                    scale_w2_mean = svd_scale[base_name + "w2"] * freq
                else:
                    scale_w2_mean += svd_scale[base_name + "w2"] * freq
                if scale_w3_mean is None:
                    scale_w3_mean = svd_scale[base_name + "w3"] * freq
                else:
                    scale_w3_mean += svd_scale[base_name + "w3"] * freq
            scale_w1_mean /= total_freq
            scale_w2_mean /= total_freq
            scale_w3_mean /= total_freq


        # self.expert_mean["w1_u"], self.expert_mean["w1_v"] = self.svd_delta(self.expert_mean["w1_mean"], ratio=self.share_ratio, svd_scale=scale_w1_mean)
        # self.expert_mean["w2_u"], self.expert_mean["w2_v"] = self.svd_delta(self.expert_mean["w2_mean"], ratio=self.share_ratio, svd_scale=scale_w2_mean)
        # self.expert_mean["w3_u"], self.expert_mean["w3_v"] = self.svd_delta(self.expert_mean["w3_mean"], ratio=self.share_ratio, svd_scale=scale_w3_mean)


        # w1_mean = self.svd_delta_reslut(self.expert_mean["w1_mean"], ratio=self.share_ratio, svd_scale=scale_w1_mean)
        # w2_mean = self.svd_delta_reslut(self.expert_mean["w2_mean"], ratio=self.share_ratio, svd_scale=scale_w2_mean)
        # w3_mean = self.svd_delta_reslut(self.expert_mean["w3_mean"], ratio=self.share_ratio, svd_scale=scale_w3_mean)

        # shared_w1_u = nn.Parameter(self.expert_mean["w1_u"])
        # shared_w1_v = nn.Parameter(self.expert_mean["w1_v"])
        # shared_w2_u = nn.Parameter(self.expert_mean["w2_u"])
        # shared_w2_v = nn.Parameter(self.expert_mean["w2_v"])
        # shared_w3_u = nn.Parameter(self.expert_mean["w3_u"])
        # shared_w3_v = nn.Parameter(self.expert_mean["w3_v"])

        delta_w1 = []
        delta_w2 = []
        delta_w3 = []

        for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):

            # self.experts[j].w1_u.weight = shared_w1_u
            # self.experts[j].w1_v.weight = shared_w1_v
            # self.experts[j].w2_u.weight = shared_w2_u
            # self.experts[j].w2_v.weight = shared_w2_v
            # self.experts[j].w3_u.weight = shared_w3_u
            # self.experts[j].w3_v.weight = shared_w3_v
            
            self.experts[j].w1.weight = shared_w1
            self.experts[j].w2.weight = shared_w2
            self.experts[j].w3.weight = shared_w3
            
            delta_w1.append(module.experts[j].w1.weight - self.expert_mean["w1_mean"])
            delta_w2.append(module.experts[j].w2.weight - self.expert_mean["w2_mean"])
            delta_w3.append(module.experts[j].w3.weight - self.expert_mean["w3_mean"])

        # Reshape delta_w1 to keep rows unchanged and concatenate columns
        # delta_w1 = torch.stack(delta_w1, dim=1).reshape(delta_w1[0].shape[0], -1)
        # delta_w2 = torch.stack(delta_w2, dim=1).reshape(delta_w2[0].shape[0], -1) 
        # delta_w3 = torch.stack(delta_w3, dim=1).reshape(delta_w3[0].shape[0], -1)
        delta_w1 = torch.stack(delta_w1, dim=0).reshape(-1, delta_w1[0].shape[1])
        delta_w2 = torch.stack(delta_w2, dim=0).reshape(-1, delta_w2[0].shape[1])
        delta_w3 = torch.stack(delta_w3, dim=0).reshape(-1, delta_w3[0].shape[1])

        if svd_scale is None:
            delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank)
            delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank)
            delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank)
        else:
            delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank, svd_scale=scale_w1_mean)
            delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank, svd_scale=scale_w2_mean)
            delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank, svd_scale=scale_w3_mean)

        shared_v1 = nn.Parameter(shared_v1)
        shared_v2 = nn.Parameter(shared_v2)
        shared_v3 = nn.Parameter(shared_v3)

        for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):

            self.experts[j].u1.weight.data = delta_u1[j * self.experts[j].u1.weight.shape[0]:(j + 1) * self.experts[j].u1.weight.shape[0], :]
            self.experts[j].v1.weight = shared_v1
            self.experts[j].u2.weight.data = delta_u2[j * self.experts[j].u2.weight.shape[0]:(j + 1) * self.experts[j].u2.weight.shape[0], :]
            self.experts[j].v2.weight = shared_v2
            self.experts[j].u3.weight.data = delta_u3[j * self.experts[j].u3.weight.shape[0]:(j + 1) * self.experts[j].u3.weight.shape[0], :]
            self.experts[j].v3.weight = shared_v3


            # print(f"w1_delta - svd w1: {delta_w1 - self.experts[j].u1.weight.data @ self.experts[j].v1.weight.data}")
            # print(f"w2_delta - svd w2: {delta_w2 - self.experts[j].u2.weight.data @ self.experts[j].v2.weight.data}")
            # print(f"w3_delta - svd w3: {delta_w3 - self.experts[j].u3.weight.data @ self.experts[j].v3.weight.data}")
            
            # print(f"w1_delta - svd result w1: {delta_w1 - self.svd_delta_reslut(delta_w1, ratio=self.delta_ratio)}")
            # print(f"w2_delta - svd result w2: {delta_w2 - self.svd_delta_reslut(delta_w2, ratio=self.delta_ratio)}")
            # print(f"w3_delta - svd result w3: {delta_w3 - self.svd_delta_reslut(delta_w3, ratio=self.delta_ratio)}")





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
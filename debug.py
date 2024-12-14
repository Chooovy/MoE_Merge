from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from evaluater import *
import torch.nn.functional as F 
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import * 
from component.merge_mixtral import *
from component.merge_mixtral_debug import *
import json
from tqdm import tqdm
import copy
# path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1/"  # 8 experts
path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"
# path = "/aifs4su/lilujun/TinyLLama-4x1.1B-MoE"

model1 = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16)

# model2 = copy.deepcopy(model1)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=5)
# ppl_eval_sharing(model1, tokenizer, experiment_name="SmolLlamix-8x101M", datasets=['wikitext2'])


# expert_freq = calculate_expert_frequency(
#     model=model1.model,
#     tokenizer=tokenizer,
#     dataset_name="wikitext",
#     split="train",
#     model_seq_len=2048,
#     batch_size=5,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     max_samples=4000,
#     seed=42,
# )


# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/TinyLLama-4x1.1B-MoE_expert_frequencies.json', 'w') as f:
#     json.dump(expert_freq, f)

# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral_expert_frequencies.json', 'r') as f:
    # expert_freq = json.load(f)

with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_expert_frequencies.json', 'r') as f:
    expert_freq = json.load(f)
# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/TinyLLama-4x1.1B-MoE_expert_frequencies.json', 'r') as f:
#     expert_freq = json.load(f)

for i in tqdm(range(len(model1.model.layers)), desc="Merging layers"):
    Merge_MoE_Block = Merge_MixtralSparseMoeBlock(model1.model.layers[i].block_sparse_moe.config, ratio=1, expert_freq=expert_freq[str(i)]).to(get_free_gpu())
    Merge_MoE_Block.merge_experts(model1.model.layers[i].block_sparse_moe)
    model1.model.layers[i].block_sparse_moe = Merge_MoE_Block



# for i in tqdm(range(len(model2.model.layers)), desc="Merging layers"):
#     Merge_MoE_Block = Merge_MixtralSparseMoeBlock_debug(model2.model.layers[i].block_sparse_moe.config, ratio=1, expert_freq=expert_freq[str(i)]).to(get_free_gpu())
#     Merge_MoE_Block.merge_experts(model2.model.layers[i].block_sparse_moe)
#     model2.model.layers[i].block_sparse_moe = Merge_MoE_Block


def calculate_nonzero_percentage(tensor, threshold=0):
    nonzero_count = torch.sum(torch.abs(tensor) > threshold)
    total_elements = tensor.numel()
    percentage = (nonzero_count / total_elements) * 100
    
    return percentage.item()


def calculate_outliers(tensor, threshold=1.5):
    """
    Calculate outliers in a tensor using IQR method.
    Args:
        tensor: Input tensor
        threshold: IQR multiplier for outlier detection (default 1.5)
    Returns:
        outlier_mask: Boolean tensor marking outliers
        outlier_percentage: Percentage of outliers
    """
    # Flatten tensor for quartile calculation
    flattened = tensor.flatten()
    
    # Calculate quartiles
    q1 = torch.quantile(flattened, 0.25)
    q3 = torch.quantile(flattened, 0.75)
    
    # Calculate IQR and bounds
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr 
    upper_bound = q3 + threshold * iqr
    
    # Create outlier mask
    outlier_mask = (tensor < lower_bound) | (tensor > upper_bound)
    
    # Calculate percentage
    outlier_percentage = (outlier_mask.sum() / tensor.numel() * 100).item()
    
    return outlier_mask, outlier_percentage



# exp_ratio = 1

# @torch.no_grad()
# def svd_delta(W, ratio=exp_ratio):
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
#     # #### Replace Attn, MLP ####
#     # sqrtSigma = torch.sqrt(truc_sigma)
#     # svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
#     # svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
#     result = truc_u @ truc_sigma @ truc_v
#     return result.to(torch.bfloat16)
#     # return 0

# # print(model)
# for i in tqdm(range(len(model2.model.layers)), desc="Merging layers"):
#     mean_w1 = None
#     mean_w2 = None
#     mean_w3 = None 
#     total_weight = 0
#     for j in range(model2.model.layers[i].block_sparse_moe.num_experts):
#         w1_weight = model2.model.layers[i].block_sparse_moe.experts[j].w1.weight
#         w2_weight = model2.model.layers[i].block_sparse_moe.experts[j].w2.weight
#         w3_weight = model2.model.layers[i].block_sparse_moe.experts[j].w3.weight
#         freq = expert_freq[str(i)][j]
#         if mean_w1 is None:
#             mean_w1 = w1_weight.clone() * freq
#         else:
#             mean_w1 += w1_weight * freq
#         if mean_w2 is None:
#             mean_w2 = w2_weight.clone() * freq
#         else:
#             mean_w2 += w2_weight * freq
#         if mean_w3 is None:
#             mean_w3 = w3_weight.clone() * freq
#         else:
#             mean_w3 += w3_weight * freq
#         total_weight += freq

#     mean_w1 /= total_weight
#     mean_w2 /= total_weight
#     mean_w3 /= total_weight

#     for j in range(model2.model.layers[i].block_sparse_moe.num_experts):
#         w1_weight = model2.model.layers[i].block_sparse_moe.experts[j].w1.weight
#         w2_weight = model2.model.layers[i].block_sparse_moe.experts[j].w2.weight
#         w3_weight = model2.model.layers[i].block_sparse_moe.experts[j].w3.weight

#         w1_weight.data.copy_(svd_delta(w1_weight - mean_w1) + mean_w1)
#         w2_weight.data.copy_(svd_delta(w2_weight - mean_w2) + mean_w2)
#         w3_weight.data.copy_(svd_delta(w3_weight - mean_w3) + mean_w3)
#     del mean_w1
#     del mean_w2
#     del mean_w3




# for i in range(len(model1.model.layers)):
#     for j in range(model1.model.layers[i].block_sparse_moe.num_experts):
#         print(calculate_nonzero_percentage(model1.model.layers[i].block_sparse_moe.experts[j].gate.weight - model2.model.layers[i].block_sparse_moe.experts[j].gate.weight))
#         print(calculate_nonzero_percentage(model1.model.layers[i].block_sparse_moe.experts[j].w1.weight - model2.model.layers[i].block_sparse_moe.experts[j].w1.weight))
#         print(calculate_nonzero_percentage(model1.model.layers[i].block_sparse_moe.experts[j].w2.weight - model2.model.layers[i].block_sparse_moe.experts[j].w2.weight))
#         print(calculate_nonzero_percentage(model1.model.layers[i].block_sparse_moe.experts[j].w3.weight - model2.model.layers[i].block_sparse_moe.experts[j].w3.weight))



for i in range(len(model1.model.layers)):
    for j in range(model1.model.layers[i].block_sparse_moe.num_experts):
        print(model1.model.layers[i].block_sparse_moe.experts[j].w1.weight)





# save_model(model, "/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_0.35_svd_delta_merged.pt")
# save_model(model, "/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral-8x7B-v0.1_0.35_svd_delta_merged.pt")
# ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=5)

ppl_eval_sharing(model1, tokenizer, experiment_name="1", datasets=['wikitext2'])
# ppl_eval_sharing(model2, tokenizer, experiment_name="2", datasets=['wikitext2'])

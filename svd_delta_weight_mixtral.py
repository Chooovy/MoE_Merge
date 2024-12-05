from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from evaluater import *
import torch.nn.functional as F 
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import * 
from tqdm import tqdm
import json

# path = "/aifs4su/lilujun/TinyLLama-4x1.1B-MoE"  # 4 experts
# path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1/"  # 8 experts

path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"

model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# expert_freq = calculate_expert_frequency(
#     model=model.model,
#     tokenizer=tokenizer,
#     dataset_name="wikitext",
#     split="train",
#     model_seq_len=2048,
#     batch_size=2,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     max_samples=4000,
#     seed=12,
# )
with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_expert_frequencies.json', 'r') as f:
    expert_freq = json.load(f)

# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral_expert_frequencies.json', 'r') as f:
#     expert_freq = json.load(f)

exp_ratio = 1

@torch.no_grad()
def svd_delta(W, ratio=exp_ratio):
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
    # #### Replace Attn, MLP ####
    # sqrtSigma = torch.sqrt(truc_sigma)
    # svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
    # svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
    result = truc_u @ truc_sigma @ truc_v
    return result.to(torch.float16)
    # return 0

# print(model)
for i in tqdm(range(len(model.model.layers)), desc="Merging layers"):
    mean_w1 = None
    mean_w2 = None
    mean_w3 = None 
    total_weight = 0
    for j in range(8):
        w1_weight = model.model.layers[i].block_sparse_moe.experts[j].w1.weight
        w2_weight = model.model.layers[i].block_sparse_moe.experts[j].w2.weight
        w3_weight = model.model.layers[i].block_sparse_moe.experts[j].w3.weight
        freq = expert_freq[str(i)][j]
        if mean_w1 is None:
            mean_w1 = w1_weight.clone() * freq
        else:
            mean_w1 += w1_weight * freq
        if mean_w2 is None:
            mean_w2 = w2_weight.clone() * freq
        else:
            mean_w2 += w2_weight * freq
        if mean_w3 is None:
            mean_w3 = w3_weight.clone() * freq
        else:
            mean_w3 += w3_weight * freq
        total_weight += freq

    mean_w1 /= total_weight
    mean_w2 /= total_weight
    mean_w3 /= total_weight

# w1不同啊
    for j in range(8):
        w1_weight = model.model.layers[i].block_sparse_moe.experts[j].w1.weight
        w2_weight = model.model.layers[i].block_sparse_moe.experts[j].w2.weight
        w3_weight = model.model.layers[i].block_sparse_moe.experts[j].w3.weight

        w1_weight.data.copy_(svd_delta(w1_weight - mean_w1) + mean_w1)
        w2_weight.data.copy_(svd_delta(w2_weight - mean_w2) + mean_w2)
        w3_weight.data.copy_(svd_delta(w3_weight - mean_w3) + mean_w3)
    del mean_w1
    del mean_w2
    del mean_w3



ppl_eval_sharing(model, tokenizer, experiment_name=f"SmolLlamix-8x101M_ratio-{exp_ratio}", datasets=['wikitext2'])


# PPL before svd: {'wikitext2': 3.8333918047584103}
# Weight Memory: 14305.17041015625 MiB

# PPL after 0.5 compression ratio svd: {'wikitext2': 6.91048967313101}
# PPL after 0.5 compression ratio svd: {'wikitext2': 6.871698553572993}
# Weight Memory: 45078.546875 MiB

# PPL after 0.25 average compression ratio svd: {'wikitext2': 13.033360644258932}
# Weight Memory: 45078.546875 MiB

# PPL after pruning: {'wikitext2': 12.885166487594889}
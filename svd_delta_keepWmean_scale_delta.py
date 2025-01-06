from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F 
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import * 
import json
from tqdm import tqdm
from functools import partial
from component.merge_mixtral_keepWmean_scale_delta import *
from component.evaluater import ppl_eval_sharing

path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1"


model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/Mixtral_wikitext_20000_expert_frequencies.json', 'r') as f:
    expert_freq = json.load(f)

svd_scale_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0-31_512.pt"
svd_scale = torch.load(svd_scale_path, map_location='cpu')

# fisher_path = "/aifs4su/lilujun/SVD-MoE-merge/outputs/fisher_Mixtral-8x7B_diagonal.pt"
fisher_path = "/aifs4su/lilujun/SVD-MoE-merge/outputs/fisher_Mixtral-8x7B.pt"
fisher_info = torch.load(fisher_path, map_location="cpu")

delta_ratio = 0.7549
share_ratio = 1
share_V = True
share_U = False
merge_method = "fisher"

layer_delta_ratio = get_rank(model, tokenizer, sparsity_ratio=delta_ratio, Hyper_m=3, Lamda=0.1, fisher_info=fisher_info)
# layer_delta_ratio = [delta_ratio] * len(model.model.layers)

for i in tqdm(range(len(model.model.layers)), desc="Merging layers"):
    Merge_MoE_Block = Merge_MixtralSparseMoeBlock(model.config, share_ratio=share_ratio, 
                                                  delta_ratio=layer_delta_ratio[i], expert_freq=expert_freq[str(i)], 
                                                  delta_share_V=share_V, delta_share_U=share_U, merge_method=merge_method).to(get_free_gpu())
    Merge_MoE_Block.merge_experts(model.model.layers[i].block_sparse_moe, svd_scale=svd_scale[i], hessian = fisher_info[i], scale_type='svdllm')
    model.model.layers[i].block_sparse_moe = Merge_MoE_Block


# sparsity_ratio = 0.4

ppl_eval_sharing(model, tokenizer, experiment_name=f"Mixtral-8x7B-delta-{delta_ratio}", 
                 datasets=['wikitext2'], params_only=False)

# save_model(model, f"/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral-8x7B-delta-{delta_ratio}-share_V-{share_V}-share_U-{share_U}-merge_method-{merge_method}-not_diagonal.pt")
import os
os.makedirs(f"/aifs4su/lilujun/SVD-MoE-merge/MoE/results/Mixtral-8x7B-delta-{delta_ratio}-share_V-{share_V}-share_U-{share_U}-merge_method-{merge_method}-owl_rank", exist_ok=True)

save_model(model, f"/aifs4su/lilujun/SVD-MoE-merge/MoE/results/Mixtral-8x7B-delta-{delta_ratio}-share_V-{share_V}-share_U-{share_U}-merge_method-{merge_method}-owl_rank/model.pt")
torch.save(layer_delta_ratio, f"/aifs4su/lilujun/SVD-MoE-merge/MoE/results/Mixtral-8x7B-delta-{delta_ratio}-share_V-{share_V}-share_U-{share_U}-merge_method-{merge_method}-owl_rank/owl_rank.pt")

# prune_wanda(model, tokenizer, nsamples=1000, seed=42, seqlen=2048, sparsity_ratio=sparsity_ratio, 
#             use_variant=True, use_rescale=False, prune_layer_name="Wmean")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from evaluater import *
import torch.nn.functional as F 
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import * 
from component.merge_mixtral import *
import json
from tqdm import tqdm

path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1/"  # 8 experts
# path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"

model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

get_expert_frequency(model, tokenizer, model_name = "Mixtral", dataset_name = "wikitext", split = "train", seed = 42, max_samples = 20000, batch_size = 32)


# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_expert_mean_freq.json', 'r') as f:
#     expert_freq = json.load(f)


# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix_wikitext_5000_expert_frequencies.json', 'r') as f:
#     expert_freq = json.load(f)


# exp_ratio = 0.35

# for i in tqdm(range(len(model.model.layers)), desc="Merging layers"):
#     Merge_MoE_Block = Merge_MixtralSparseMoeBlock(model.model.layers[i].block_sparse_moe.config, ratio=exp_ratio, expert_freq=expert_freq[str(i)]).to(get_free_gpu())
#     Merge_MoE_Block.merge_experts(model.model.layers[i].block_sparse_moe)
#     model.model.layers[i].block_sparse_moe = Merge_MoE_Block


# ppl_eval_sharing(model, tokenizer, experiment_name=f"SmolLlamix-8x101M_ratio-{exp_ratio}", datasets=['wikitext2'], params_only=False)

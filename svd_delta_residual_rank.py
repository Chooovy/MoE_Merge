from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import torch.nn.functional as F 
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import * 
import json
from tqdm import tqdm
from functools import partial
from component.merge_mixtral_residual_rank import *
from component.evaluater import ppl_eval_sharing

path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1/"  # 8 experts
# path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"
# path = "/aifs4su/gov/models/Llama-2-7b-chat-hf/"


model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# svd_scale = get_svd_scale(model, tokenizer, "Mixtral-8x7B-v0.1", max_samples=1000)
# torch.save(svd_scale, "/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral.pt")
# ppl_eval_sharing(model, tokenizer, experiment_name="SmolLlamix-8x101M", datasets=['wikitext2'], params_only=False)

# expert_outputs = calculate_expert_outputs(
#     model=model.model,
#     tokenizer=tokenizer,
#     max_samples=1000  # 限制样本数量
# )

# Save expert outputs to .pt file
# torch.save(expert_outputs, '/aifs4su/lilujun/SVD-MoE-merge/MoE/expert_outputs.pt')

# Load expert outputs from .pt file 
# expert_outputs = torch.load('/aifs4su/lilujun/SVD-MoE-merge/MoE/expert_outputs.pt')


# expert_freq = calculate_expert_frequency(
#     model=model.model,
#     tokenizer=tokenizer,
#     dataset_name="wikitext",
#     split="train",
#     model_seq_len=2048,
#     batch_size=5,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     max_samples=4000,
#     seed=42,
# )


# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_expert_frequencies.json', 'w') as f:
#     json.dump(expert_freq, f)

with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/Mixtral_wikitext_20000_expert_frequencies.json', 'r') as f:
    expert_freq = json.load(f)

# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SmolLlamix_wikitext_5000_expert_frequencies.json', 'r') as f:
#     expert_freq = json.load(f)

# svd_scale_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_SmolLlamix.pt"
# svd_scale = torch.load(svd_scale_path)

# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_expert_mean_freq.json', 'r') as f:
#     expert_freq = json.load(f)

delta_ratio = 0.25
share_ratio = 1

for i in tqdm(range(len(model.model.layers)), desc="Merging layers"):
    Merge_MoE_Block = Merge_MixtralSparseMoeBlock(model.model.layers[i].block_sparse_moe.config, share_ratio=share_ratio, delta_ratio=delta_ratio, expert_freq=expert_freq[str(i)]).to(get_free_gpu())
    Merge_MoE_Block.merge_experts(model.model.layers[i].block_sparse_moe, svd_scale=None)
    model.model.layers[i].block_sparse_moe = Merge_MoE_Block



# save_model(model, "/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_0.35_svd_delta_merged.pt")
# save_model(model, "/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral-8x7B-v0.1_0.35_svd_delta_merged.pt")
# ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=5)

ppl_eval_sharing(model, tokenizer, experiment_name=f"Mixtral-8x7B-v0.1-delta-{delta_ratio}-share-{share_ratio}", datasets=['wikitext2'], params_only=False)

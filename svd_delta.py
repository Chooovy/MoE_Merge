from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from evaluater import ppl_eval
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
# config = AutoConfig.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# print(model)

print(f"Total parameters of base model: {count_parameters(model)}")
ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=5, device="cuda")


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

with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral_expert_frequencies.json', 'r') as f:
    expert_freq = json.load(f)

# with open('/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_expert_frequencies.json', 'r') as f:
#     expert_freq = json.load(f)


for i in tqdm(range(len(model.model.layers)), desc="Merging layers"):
    Merge_MoE_Block = Merge_MixtralSparseMoeBlock(model.model.layers[i].block_sparse_moe.config, ratio=0.35, expert_freq=expert_freq[str(i)]).to(get_free_gpu())
    Merge_MoE_Block.merge_experts(model.model.layers[i].block_sparse_moe)
    model.model.layers[i].block_sparse_moe = Merge_MoE_Block



# save_model(model, "/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_0.5_svd_delta_merged.pt")
# save_model(model, "/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral-8x7B-v0.1_svd_delta_merged.pt")


print(f"Total parameters after merge: {count_parameters(model)}")
# 12873633792
# 46702792704
# 24141631488
ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=5, device="cuda")

# PPL before svd: {'wikitext2': 3.8333918047584103}
# Weight Memory: 14305.17041015625 MiB

# PPL after 0.5 compression ratio svd: {'wikitext2': 6.91048967313101}
# Weight Memory: 45078.546875 MiB

# PPL after 0.25 average compression ratio svd: {'wikitext2': 13.033360644258932}
# Weight Memory: 45078.546875 MiB
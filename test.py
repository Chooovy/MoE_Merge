from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from evaluater import ppl_eval
import torch.nn.functional as F 
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import * 
from component.merge_mixtral import *
import json
from tqdm import tqdm


path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"  # 8 experts

model = load_model(path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_0.5_svd_delta_merged.pt")
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

print(model)

print(f"Total parameters after merge: {count_parameters(model)}")

ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=5, device="cuda")
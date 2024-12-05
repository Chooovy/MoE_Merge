from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, default_data_collator
import torch
from evaluater import *
import torch.nn.functional as F 
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import * 
from component.merge_mixtral import *
import json
from tqdm import tqdm
from accelerate import Accelerator

# path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"  # 8 experts
base_model_path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1/"  # 8 experts
# checkpoint_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_0.35_svd_delta_merged.pt"
checkpoint_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral-8x7B_0.0_svd_delta_merged.pt"

model = load_model_tqdm(checkpoint_path = checkpoint_path, base_model_path=base_model_path, ratio=0)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16)

ppl_eval_sharing(model, tokenizer, experiment_name="SmolLlamix-8x101M", datasets=['wikitext2'])
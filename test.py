from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import torch.nn.functional as F 
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import * 
import json
from tqdm import tqdm
from functools import partial
from component.merge_mixtral_keepWmean_scale_delta import *
from component.evaluater import ppl_eval_sharing

# path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"  # 8 experts
base_model_path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1/"  # 8 experts

checkpoint_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral-8x7B-delta-0.5-share-1-share_V-False-share_U-False.pt"

def load_model_tqdm(checkpoint_path, base_model_path, delta_ratio = 0.5, share_ratio = 1):
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(base_model_path, 
                                                    device_map="auto", 
                                                    trust_remote_code=True, 
                                                    torch_dtype=torch.bfloat16)
        
    
    for i in tqdm(range(len(model.model.layers)), desc="Initializing layers"):
        model.model.layers[i].block_sparse_moe = Merge_MixtralSparseMoeBlock(model.model.layers[i].block_sparse_moe.config, 
                                                                             share_ratio=share_ratio, 
                                                                             delta_ratio=delta_ratio, 
                                                                             expert_freq=None).to(model.model.layers[i].block_sparse_moe.gate.weight.device)

    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

    pbar = tqdm(total=len(checkpoint), desc="Loading checkpoint")
    for k, v in checkpoint.items():
        model.state_dict()[k].copy_(v)
        pbar.update(1)
    pbar.close()
    
    return model

model = load_model_tqdm(checkpoint_path = checkpoint_path, base_model_path=base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.bfloat16)

ppl_eval_sharing(model, tokenizer, experiment_name="Mixtral-8x7B-delta-0.5-share-1", datasets=['wikitext2'])
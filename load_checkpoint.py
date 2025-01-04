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

cfg = {}

cfg['merge_model'] = True
cfg['deepcopy'] = False
using_Mixtral = False

if cfg['merge_model']:
    if using_Mixtral:
        base_model_path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1/"
    else:
        base_model_path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"

    if using_Mixtral:
        checkpoint_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/Mixtral-8x7B-delta-0.5-share_V-True-share_U-False-merge_method-fisher.pt"
    else:
        if cfg['deepcopy']:
            checkpoint_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M-delta-0.5-share_V-True-share_U-False-deepcopy.pt"
        else:
            checkpoint_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M-delta-0.5-share_V-True-share_U-False.pt"

    def load_model_tqdm(checkpoint_path, base_model_path, delta_ratio = 0.5, share_ratio = 1, share_V = True, share_U = False, merge_method = "fisher"):
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(base_model_path, 
                                                        device_map="auto", 
                                                        trust_remote_code=True, 
                                                        torch_dtype=torch.bfloat16)
            
        
        for i in tqdm(range(len(model.model.layers)), desc="Initializing layers"):
            model.model.layers[i].block_sparse_moe = Merge_MixtralSparseMoeBlock(model.config, share_ratio=share_ratio, 
                                                delta_ratio=delta_ratio, expert_freq=None, 
                                                delta_share_V=share_V, delta_share_U=share_U, 
                                                merge_method=merge_method, deepcopy=cfg['deepcopy']).to(model.model.layers[i].block_sparse_moe.gate.weight.device)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        pbar = tqdm(total=len(checkpoint), desc="Loading checkpoint")
        for k, v in checkpoint.items():
            # model.state_dict()[k] = v.to(model.state_dict()[k].device)
            model.state_dict()[k].copy_(v)
            pbar.update(1)
        pbar.close()
        
        return model

    model = load_model_tqdm(checkpoint_path = checkpoint_path, base_model_path=base_model_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
else:
    if using_Mixtral:
        path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1"
    else:
        path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# -----------------------------------------------------------------------------------------

ppl_eval_sharing(model, tokenizer, experiment_name="SmolLlamix", datasets=['wikitext2'])
import random
import numpy as np
from transformers.models.mixtral.modeling_mixtral import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path
import json
from tqdm import tqdm
import os
from accelerate import init_empty_weights
from functools import partial
# from .merge_mixtral import Merge_MixtralSparseMoeBlock


def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    return total_params

def should_process_layer(name, model_name, Attn_or_Experts, layer_idx, attention_layers, expert_layers, module, for_expert_counting=False):
    """Helper function to determine if a layer should be processed"""
    if isinstance(module, (MixtralSparseMoeBlock)):
        return for_expert_counting  # 仅在统计专家选择时处理 MoE 模块
    else:
        if for_expert_counting:
            return False  # 非 MoE 模块不需要用于专家选择计数
        else:
            # 判断是否需要收集 scaling_matrix
            if 'Mixtral' in model_name or 'deepseek' in model_name or 'phimoe' in model_name:
                if Attn_or_Experts == 'Attn' and 'experts' in name:
                    return False
                if Attn_or_Experts == 'Experts' and 'experts' not in name:
                    return False
                if 'experts' in name and (layer_idx is None or layer_idx not in expert_layers):
                    return False
                if 'attn' in name and (layer_idx is None or layer_idx not in attention_layers):
                    return False
                if 'Mixtral' in model_name and 'gate' in name:
                    return False
                if 'deepseek' in model_name and type(module).__name__ == 'MoEGate':
                    return False
                if 'phimoe' in model_name and 'block_sparse_moe.gate' in name:
                    return False
            return True  # 符合条件的模块，收集 scaling_matrix


def find_layers(module, layers=[nn.Conv2d, nn.Linear, MixtralSparseMoeBlock], name='', process_moe_block=False):
    res = {}

    if isinstance(module, MixtralSparseMoeBlock) or type(module).__name__ == 'MoEGate' or type(module).__name__ == 'PhiMoESparseMoeBlock':
        if process_moe_block:
            res[name] = module
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=process_moe_block
                ))
        else:
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=False
                ))
        return res 
    elif type(module) in layers or 'gate' in name:
        res[name] = module
    else:
        for name1, child in module.named_children():
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1, process_moe_block=process_moe_block
            ))

    return res



def find_linear_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res





def save_model(model, path):
    """Saves the model to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")



def load_model_tqdm(checkpoint_path, base_model_path, ratio=0.35):
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(base_model_path, 
                                                    device_map="auto", 
                                                    trust_remote_code=True, 
                                                    torch_dtype=torch.bfloat16)
        
    
    for i in tqdm(range(len(model.model.layers)), desc="Initializing layers"):
        model.model.layers[i].block_sparse_moe = Merge_MixtralSparseMoeBlock(
            model.model.layers[i].block_sparse_moe.config, 
            ratio=ratio, 
            expert_freq=None
        ).to(model.model.layers[i].block_sparse_moe.gate.weight.device)
    
    # print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    
    # print("Loading state dict...")
    # ?????????????????????????
    pbar = tqdm(total=len(checkpoint), desc="Loading checkpoint")
    for k, v in checkpoint.items():
        model.state_dict()[k].copy_(v)
        pbar.update(1)
    pbar.close()
    
    return model


def set_seed(seed):

    seed = 42  # or any other integer
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

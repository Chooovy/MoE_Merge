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

path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1/"  # 8 experts
# path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"
# path = "/aifs4su/gov/models/Llama-2-7b-chat-hf/"


model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16)


tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--selected_layers', type=str, default='',help='Comma separated list of layer indices to select')
args = parser.parse_args()
selected_layers = [int(x) for x in args.selected_layers.split(',')]

num_samples = 512

svd_scale = get_svd_scale(model, tokenizer, "Mixtral-8x7B-v0.1", max_samples=num_samples, selected_layers=selected_layers, dataset_name="wikitext")

selected_layers_str = '_'.join(map(str, selected_layers))

torch.save(svd_scale, f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_{selected_layers_str}_{num_samples}.pt")

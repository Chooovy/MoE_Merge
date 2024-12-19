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
base_model_path = "/workspace/SVD-MOE-new/models_Mixtral"  # 8 experts
# checkpoint_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_0.35_svd_delta_merged.pt"
checkpoint_path = "/workspace/guhao_workspace/MoE_Merge/Mixtral-8x7B-delta-0.5-share-1-share_V-False-share_U-False.pt"

def load_model_tqdm(checkpoint_path, base_model_path, delta_ratio = 0.5, share_ratio = 1, share_V = False, share_U = False):
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(base_model_path, 
                                                    device_map="auto", 
                                                    trust_remote_code=True, 
                                                    torch_dtype=torch.bfloat16)
        
    
    for i in tqdm(range(len(model.model.layers)), desc="Initializing layers"):
        model.model.layers[i].block_sparse_moe = Merge_MixtralSparseMoeBlock(model.model.layers[i].block_sparse_moe.config, 
                                                                             share_ratio=share_ratio, 
                                                                             delta_ratio=delta_ratio, 
                                                                             expert_freq=None,
                                                                             delta_share_V=share_V, 
                                                                             delta_share_U=share_U).to(model.model.layers[i].block_sparse_moe.gate.weight.device)

    checkpoint = torch.load(checkpoint_path)

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


ppl_eval_sharing(model, tokenizer, experiment_name="Mixtral-8x7B-delta-0.5-share-1", datasets=['wikitext2'])

sparsity_ratio = 0.2
prune_wanda(model, tokenizer, nsamples=128, seed=42, seqlen=2048, prune_n=4, prune_m=8, 
            sparsity_ratio=sparsity_ratio, use_variant=True, use_rescale=False, prune_layer_name="Wmean")

ppl_eval_sharing(model, tokenizer, experiment_name="Mixtral-8x7B-delta-0.5-share-1", datasets=['wikitext2'])


max_similarity = -float('inf')
most_similar_pair = None
for m in range(32):  
    for n in range(m,32):
        for j in range(8):
            similarity = linear_cka(model.model.layers[m].block_sparse_moe.experts[j].Wmean2, 
                                 model.model.layers[n].block_sparse_moe.experts[j].Wmean2)
            if similarity > max_similarity and m != n and similarity != 1.0:
                max_similarity = similarity
                most_similar_pair = (m, n)




max_similarity = -float('inf')
most_similar_pair = None
for m in range(32):
    # print(f"layer {m}")
    for i in range(8):
        for j in range(i, 8):
            print(f"layer {m} expert {i} and {j}")
            similarity = linear_cka(model.model.layers[m].block_sparse_moe.experts[i].delta_v1, 
                                    model.model.layers[m].block_sparse_moe.experts[j].delta_v1)
            # if similarity > max_similarity and i != j and similarity != 1.0:
            #     max_similarity = similarity
            #     most_similar_pair = (m, i, j)        
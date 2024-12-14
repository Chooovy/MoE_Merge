from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from evaluater import *
import torch.nn.functional as F 
from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.qwen2_moe.modeling_qwen2_moe import * 
from component.merge_mixtral import *
import json
from tqdm import tqdm
from functools import partial

# path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1/"  # 8 experts
path = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"

model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

selected_layers = list(range(len(model.model.layers)))
model_name = "SmolLlamix"

device = next(model.parameters()).device
expert_selection_counts = {i: torch.zeros(8, device=device) for i in selected_layers}

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
dataset = dataset.select(range(1000))
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


def hook_for_expert_counting(module, input, output, module_name=None):
    if isinstance(module, MixtralSparseMoeBlock):
        router_logits = output[1]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1)

        for expert_idx in selected_experts.unique():
            if expert_idx.item() < len(layer_expert_selection_counts):
                layer_expert_selection_counts[expert_idx.item()] += (selected_experts == expert_idx).sum().item()
            else:
                logger.warning(f"Expert index {expert_idx.item()} out of range")
        return


# Iterate through your data loader
for batch in tqdm(dataloader, desc="Collecting expert outputs"):
    # Forward pass
    inputs = tokenizer(
        batch['text'],
        truncation=True,
        padding=True,
        max_length=2048,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 前向传播
    with torch.no_grad():
        model(**inputs)
    
    # for layer_idx, layer in enumerate(model.model.layers):
    #     for module in layer.modules():
    #         if isinstance(module, MixtralSparseMoeBlock):
    #             # Ensure expert_frequency is a list and convert to tensor
    #             expert_freq_tensor = torch.tensor(module.expert_frequency, device=device)
    #             expert_selection_counts[layer_idx] += expert_freq_tensor
    #             break  # Assuming only one MixtralSparseMoeBlock per layer
    
    for layer_idx, layer in enumerate(model.model.layers):
        for module in layer.modules():
            if isinstance(module, MixtralSparseMoeBlock):
                router_logits = output[1]
                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1)
                for expert_idx in selected_experts.unique():
                    if expert_idx.item() < 8:
                        layer_expert_selection_counts[expert_idx.item()] += (selected_experts == expert_idx).sum().item()

    # Optionally reset expert_frequency if it's not cumulative
    # for layer_idx, layer in enumerate(model.model.layers):
    #     for module in layer.modules():
    #         if isinstance(module, MixtralSparseMoeBlock):
    #             module.expert_frequency = [0] * config.num_local_experts
    #             break

# After all batches
for layer, counts in expert_selection_counts.items():
    print(f"Layer {layer}: {counts.tolist()}")
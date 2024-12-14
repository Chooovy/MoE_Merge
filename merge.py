import torch
from torch.utils.data import DataLoader
from torch.nn import ModuleList
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from component.evaluater import ppl_eval_sharing

# Step 1: Load the MoE model and identify experts
model_name = "/aifs4su/lilujun/SVD-MoE-merge/SmolLlamix-8x101M"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Access the MoE layer (adjust indices and attribute names as necessary)
moe_layers = model.model.layers
layer_idx = 0  # Adjust based on the specific MoE layer
moe_layer = moe_layers[layer_idx]
experts = moe_layer.block_sparse_moe.experts  # Adjust attribute names if needed
num_experts = len(experts)
print(f"Number of experts: {num_experts}")

# Step 2: Compute the Fisher Information Matrices for each expert
def compute_fisher_for_expert(expert, dataloader, device='cuda'):
    expert.to(device)
    expert.train()
    fisher_dict = {name: torch.zeros_like(param) for name, param in expert.named_parameters() if param.requires_grad}
    criterion = torch.nn.CrossEntropyLoss()
    num_batches = 0
    for batch in dataloader:
        num_batches += 1
        expert.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        inputs = batch['input_ids']
        outputs = expert(inputs)
        logits = outputs.logits
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)
        loss = criterion(logits, targets)
        loss.backward()
        for name, param in expert.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2)
    for name in fisher_dict:
        fisher_dict[name] /= num_batches
    expert.to('cpu')
    return fisher_dict

# Prepare dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_dataset.set_format(type='torch')
dataloader = DataLoader(tokenized_dataset, batch_size=8)

# dataloader = torch.load("/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/dataloader_wikitext2_128.pt")

# Compute Fisher matrices
expert_fishers = []
for idx, expert in enumerate(experts):
    print(f"Computing Fisher matrix for expert {idx+1}/{num_experts}")
    fisher = compute_fisher_for_expert(model, dataloader)
    expert_fishers.append(fisher)

# Step 3: Merge the experts
def merge_experts(experts, expert_fishers, scaling_factors, delta_0, scaling_factor_overall):
    merged_params = {}
    preconditioners = {}
    param_names = experts[0].state_dict().keys()
    for name in param_names:
        summed_fisher = sum(
            scaling_factor * expert_fisher[name]
            for expert_fisher, scaling_factor in zip(expert_fishers, scaling_factors)
        )
        preconditioner = 1.0 / (summed_fisher + delta_0)
        preconditioners[name] = preconditioner
    for name in param_names:
        param_diffs = sum(
            scaling_factor * (expert.state_dict()[name] - experts[0].state_dict()[name])
            for expert, scaling_factor in zip(experts, scaling_factors)
        )
        merged_param = experts[0].state_dict()[name] + scaling_factor_overall * preconditioners[name] * param_diffs
        merged_params[name] = merged_param
    merged_expert = type(experts[0])()
    merged_expert.load_state_dict(merged_params)
    return merged_expert

delta_0 = 1e-12
scaling_factor_overall = 1.0
scaling_factors = [1.0 for _ in experts]

merged_expert = merge_experts(experts, expert_fishers, scaling_factors, delta_0, scaling_factor_overall)

# Step 4: Update the MoE model
moe_layer.block_sparse_moe.experts = ModuleList([merged_expert])

# Step 5: Validate and save the updated model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
input_text = "Once upon a time, in a land far away"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Text:")
print(output_text)
ppl_eval_sharing(model, tokenizer, experiment_name="merge_SmolLlamix-8x101M", datasets=['wikitext2'], params_only=False)

# Save the updated model
# output_model_path = 'merged_SmolLlamix-8x101M'
# model.save_pretrained(output_model_path)
# tokenizer.save_pretrained(output_model_path)
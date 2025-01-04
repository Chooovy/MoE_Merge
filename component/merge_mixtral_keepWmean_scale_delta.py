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
import random
from .model_utils import find_layers, find_linear_layers
from .data_utils import get_loaders
import numpy as np

def save_model(model, path):
    """Saves the model to the specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def linear_cka(Linear1, Linear2):
    # 将两个权重矩阵移到同一个设备上
    device = Linear1.weight.device
    W1 = Linear1.weight.data.to(device)
    W2 = Linear2.weight.data.to(device)
    
    # 中心化
    def center_kernel(K):
        n = K.size(0)
        one_n = (torch.ones(n, n, dtype=torch.bfloat16) / n).to(device)
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # 计算Gram矩阵
    K = W1 @ W1.T
    L = W2 @ W2.T

    # 中心化Gram矩阵
    Kc = center_kernel(K)
    Lc = center_kernel(L)

    # 计算CKA
    numerator = torch.trace(Kc @ Lc)
    denominator = torch.sqrt(torch.trace(Kc @ Kc) * torch.trace(Lc @ Lc))
    cka_score = numerator / denominator
    print(f"CKA between linear1 and linear2: {cka_score.item():.4f}")
    return cka_score.item()


def weight_cka(W1: torch.Tensor, W2: torch.Tensor) -> float:
    # 中心化
    def center_kernel(K):
        n = K.size(0)
        one_n = (torch.ones(n, n, dtype=torch.bfloat16) / n).to(K.device)
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    # 计算Gram矩阵
    K = W1 @ W1.T
    L = W2 @ W2.T

    # 中心化Gram矩阵
    Kc = center_kernel(K)
    Lc = center_kernel(L)

    # 计算CKA
    numerator = torch.trace(Kc @ Lc)
    denominator = torch.sqrt(torch.trace(Kc @ Kc) * torch.trace(Lc @ Lc))
    cka_score = numerator / denominator
    print(f"CKA between linear1 and linear2: {cka_score.item():.4f}")


def cal_scale_inv(svd_scale):
    try:
        scale_inv = torch.linalg.inv(svd_scale.to(torch.float32))
    except Exception as e:
        print("Warning: svd_scale is not full rank!")
        svd_scale += 1e-6 * torch.eye(svd_scale.shape[0]).to(svd_scale.device)
        scale_inv = torch.linalg.inv(svd_scale)
    return scale_inv.float()

def get_free_gpu():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return 'cpu'
    free_memory = [torch.cuda.mem_get_info(i)[0] for i in range(num_gpus)]
    most_free_gpu_index = int(torch.argmax(torch.tensor(free_memory)))
    return f'cuda:{most_free_gpu_index}'

def get_expert_frequency(model, tokenizer, model_name, dataset_name, split, seed, max_samples = None, batch_size = 32):
    selected_layers = list(range(len(model.model.layers)))

    device = next(model.parameters()).device
    expert_selection_counts = {i: torch.zeros(model.model.config.num_local_experts, device=device) for i in selected_layers}

    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        text_column = 'text'
    elif 'ptb' in dataset_name:
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split=split)
        text_column = 'sentence'
    elif 'c4' in dataset_name:
        dataset = load_dataset("json", data_files="/aifs4su/lilujun/SVD-MoE-merge/SVD-MOE-new/function_base/c4-train.json")[split]
        text_column = 'text'
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(max_samples, len(dataset))))


    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    handles = []

    def hook_for_expert_counting(module, input, output, module_name=None):
        if isinstance(module, MixtralSparseMoeBlock):
            router_logits = output[1]  # Assuming the router logits are the second output
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, selected_experts = torch.topk(routing_weights, k=module.top_k, dim=-1)

            for expert_idx in selected_experts.unique():
                if expert_idx.item() < expert_selection_counts[module_name].size(0):
                    expert_selection_counts[module_name][expert_idx.item()] += (selected_experts == expert_idx).sum().item()
                else:
                    logger.warning(f"Expert index {expert_idx.item()} out of range for module {module_name}")

    def create_hook(layer_idx):
        """
        Creates a partial hook function for a specific layer.
        """
        return partial(hook_for_expert_counting, module_name=layer_idx)

    # Register hooks for each expert in each selected layer
    for layer_idx in selected_layers:
        layer = model.model.layers[layer_idx]
        if hasattr(layer, 'block_sparse_moe'):
            moe_module = layer.block_sparse_moe
            handle = moe_module.register_forward_hook(create_hook(layer_idx))
            handles.append(handle)
        # elif hasattr(layer, 'gate'):
        #     gate_module = layer.gate
        #     handle = gate_module.register_forward_hook(create_hook(layer_idx))
        #     handles.append(handle)

    # Iterate through the dataloader and perform forward passes to collect counts
    for batch in tqdm(dataloader, desc="Collecting expert activation counts"):
        inputs = tokenizer(batch[text_column], truncation=True, padding=True, max_length=2048, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    # Remove all hooks after collection
    for handle in handles:
        handle.remove()

    # Save the counts to a JSON file
    counts_dict = {layer: counts.tolist() for layer, counts in expert_selection_counts.items()}

    with open(f"/aifs4su/lilujun/SVD-MoE-merge/MoE/{model_name}_{dataset_name}_{max_samples}_expert_frequencies.json", "w") as f:
        json.dump(counts_dict, f, indent=4)




def process_scaling_matrix(raw_matrix, name, module_device):
    """Helper function to process scaling matrix"""
    try:
        if isinstance(raw_matrix, (float, int)):
            return raw_matrix

        matrix = raw_matrix.clone()

        # Ensure the matrix is symmetric
        matrix = (matrix + matrix.T) / 2

        # min_eigenval = torch.linalg.eigvalsh(matrix)[0].item()
        # if min_eigenval <= 0:
        #     adjustment = (-min_eigenval + 1e-6)
        #     matrix += adjustment * torch.eye(
        #         matrix.shape[0],
        #         device=module_device,
        #         dtype=matrix.dtype
        #     )
        # del min_eigenval

        cholesky_matrix = torch.linalg.cholesky(matrix)
        return cholesky_matrix

    except Exception as e:
        logger.error(f"Error processing scaling matrix for {name}: {str(e)}")

        # Attempt to adjust the matrix further
        epsilon = 1e-6
        max_attempts = 20
        attempt = 0

        while attempt < max_attempts:
            try:
                # Increase the adjustment
                matrix += epsilon * torch.eye(
                    matrix.shape[0],
                    device=module_device,
                    dtype=matrix.dtype
                )
                epsilon *= 10
                attempt += 1

                # Try Cholesky decomposition again
                cholesky_matrix = torch.linalg.cholesky(matrix)
                return cholesky_matrix

            except Exception as e_inner:
                # Log the attempt
                logger.error(f"Attempt {attempt}: Failed to process scaling matrix for {name}: {str(e_inner)}")

        # If all attempts fail, fall back to Eigen decomposition
        logger.error(f"All attempts failed for {name}. Using Eigen decomposition as fallback.")
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        # Ensure eigenvalues are positive
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        adjusted_matrix = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        # Perform Cholesky decomposition on the adjusted matrix
        cholesky_matrix = torch.linalg.cholesky(adjusted_matrix)
        return cholesky_matrix

@torch.no_grad()
def get_svd_scale(model, tokenizer, model_name, dataset_name='wikitext', split='train', 
                    seed=42, seqlen=2048, batch_size=1, max_samples=None, selected_layers=None):

    layers = model.model.layers
    layers[0] = layers[0]

    cache_file = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/calib_loader_{model_name}_{dataset_name}_{max_samples}.pt"

    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        text_column = 'text'
        tot_text = "\n\n".join(dataset[text_column])
    elif 'ptb' in dataset_name:
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split=split)
        text_column = 'sentence'
        tot_text = "\n\n".join(dataset[text_column])
    elif 'c4' in dataset_name:
        dataset = load_dataset("json", data_files="/aifs4su/lilujun/SVD-MoE-merge/SVD-MOE-new/function_base/c4-train.json")[split]
        text_column = 'text'
        tot_text = "\n\n".join(dataset[text_column])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    

    calib_loader = []
    for s in range(max_samples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        if trainenc.input_ids.shape[1] < seqlen:
            s = s - 1
            continue
        if s % batch_size == 0:
            if s != 0:
                attention_mask = torch.ones_like(inp)
                calib_loader.append({"input_ids": inp, "attention_mask": attention_mask})
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)

    # torch.save(calib_loader, cache_file)

    # calib_loader = torch.load(cache_file)

    if selected_layers is None:
        selected_layers = list(range(len(layers)))
    else:
        selected_layers = selected_layers
    

    dtype = next(iter(model.parameters())).dtype
    device = get_free_gpu()

    inps = torch.zeros(
        (len(calib_loader), seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.detach()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].detach()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].detach()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])

    for batch in calib_loader:
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    
    layers[0] = layers[0].module


    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids'].detach()
    profiling_mat = {}
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        process_subset = {}
        layer = layers[i]
        # subset = find_layers(module = layer, layers=[nn.Linear, MixtralSparseMoeBlock], process_moe_block=True) 
        if i in selected_layers:
            subset = find_linear_layers(module = layer, layers=[nn.Linear])

            for name, module in subset.items():
                if 'experts' in name:
                    process_subset[name] = module

        def hook(module, input, output):
            module_device = next(module.parameters()).device
            inp = input[0].detach().float().to(module_device)
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            inp = inp.view(-1, inp.size(-1))
            adds = torch.matmul(inp.transpose(0, 1), inp)

            # min_eigenval = torch.linalg.eigvalsh(adds)[0].item()
            # if min_eigenval < 0:
            #     adjustment = (-min_eigenval + 1e-6)
            #     adds += adjustment * torch.eye(adds.shape[0], device=module_device, dtype=torch.bfloat16)

            module.scaling_diag_matrix += adds
            del inp, adds, output
            torch.cuda.empty_cache()
        
        handles = []
        
        for name in process_subset:
            process_subset[name].scaling_diag_matrix = 0
            handles.append(process_subset[name].register_forward_hook(hook))

        for j in range(inps.shape[0]):
            if attention_masks is not None:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(device))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()
        for name in process_subset:
            device = get_free_gpu()
            raw_scaling_diag_matrix = process_subset[name].scaling_diag_matrix.double().to(device)
            try:
                scaling_diag_matrix = process_scaling_matrix(raw_scaling_diag_matrix, name, device)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(device)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.to(torch.bfloat16)
            scaling_diag_matrix = raw_scaling_diag_matrix = process_subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, process_subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer
        profiling_mat[i] = layer_profile
        inps = outs
        torch.cuda.empty_cache()
    return profiling_mat



def prepare_calibration_input(model, dataloader, device, seqlen):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # Calculate total number of samples from dataloader
    total_samples = sum(1 for _ in dataloader)
    inps = torch.zeros((total_samples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids



class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples




def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity




def prune_wanda(model, tokenizer, device=None, prune_n=0, prune_m=0, nsamples=128, seed=0, seqlen=2048, 
                use_variant = False, sparsity_ratio = 0.5, use_rescale = False, prune_layer_name = "Wmean"):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer)
    # torch.save(dataloader, "/workspace/guhao_workspace/MoE_Merge/cache/dataloader_wikitext2_128.pt")
    # dataloader = torch.load("/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/dataloader_wikitext2_128.pt")
    print("dataset loading complete")
    device = get_free_gpu()
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, seqlen=seqlen)

    layers = model.model.layers
    # for i in range(len(layers)):
    for i in tqdm(range(len(layers)), desc="Pruning layers"):
        layer = layers[i]
        subset = find_linear_layers(module = layer, layers=[nn.Linear])

        process_subset = {}
        for name, module in subset.items():
            if prune_layer_name in name:
                process_subset[name] = module

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            if attention_mask is not None and position_ids is not None:
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            else:
                inps, outs = inps.to(dev), outs.to(dev)

        wrapped_layers = {}
        for name in process_subset:
            wrapped_layers[name] = WrappedGPT(process_subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(process_subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in process_subset:
            # print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(process_subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                # print(f"{prune_n} : {prune_m} structured pruning")
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    # print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    # print(f"unstructured pruning")
                    indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            process_subset[name].weight.data[W_mask] = 0  ## set weights to zero
            if use_rescale and sparsity_ratio != 1.0:
                process_subset[name].weight.data = process_subset[name].weight.data / (1 - sparsity_ratio)
            # process_subset[name].register_buffer('W_mask', W_mask)


        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()



def find_layers_rank(module, layers=[nn.Linear], name='', include_name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.
        include_name (str): Only include layers whose names contain this string.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        if include_name == '' or include_name in name:
            return {name: module}
        return {}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers_rank(
            child, 
            layers=layers, 
            name=name + '.' + name1 if name != '' else name1,
            include_name=include_name
        ))
    return res

def check_outlier_mean(mask,threshold):
    W = mask
    count = 0 
    total_params = 0
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()
    outlier_ratio=float(count)/total_params*100
    return outlier_ratio

def get_rank(model, tokenizer, device=torch.device("cuda:0"), nsamples = 128, seed = 42, seqlen = 2048, 
                        Hyper_m = 5, Lamda = 0.08, sparsity_ratio = 0.5, fisher_info = None):
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    # print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2",nsamples=nsamples,seed=seed,seqlen=seqlen,tokenizer=tokenizer)
    # print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, seqlen)
    # print ("inps",inps)

    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers_rank(layer, include_name="experts")
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev) if inps is not None else None
            outs = outs.to(dev) if outs is not None else None 
            attention_mask = attention_mask.to(dev) if attention_mask is not None else None
            position_ids = position_ids.to(dev) if position_ids is not None else None
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
    
        layer_wmetric=[]

        for name in subset:
            # print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            if fisher_info is not None:
                W_metric = W_metric * fisher_info[i][name]

            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)    
                
        for j in range(nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [Hyper_m]:
            
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            # print ("layer outlier ratio",out_ratio,out_ratio_layer)

        all_layer_ratio.append(out_ratio_layer)

    # print ("before adjustment",all_layer_ratio)
    
    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * Lamda*2))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-sparsity_ratio)
    # print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))

    print ("after adjustment",all_layer_ratio  )
    return all_layer_ratio




class Merge_MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config, share_ratio, delta_ratio, expert_freq, delta_share_V=False, delta_share_U=False, merge_method="freq", deepcopy = False):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.intermediate_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.dtype = torch.bfloat16

        # added
        self.share_ratio = share_ratio
        self.delta_ratio = delta_ratio
        self.expert_freq = expert_freq
        self.config = config

        self.delta_share_V = delta_share_V
        self.delta_share_U = delta_share_U
        self.merge_method = merge_method
        self.deepcopy = deepcopy

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False, dtype=torch.bfloat16)
        
        self.Wmean1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)
        self.Wmean2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=self.dtype)
        self.Wmean3 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)

        nn.init.zeros_(self.Wmean1.weight)
        nn.init.zeros_(self.Wmean2.weight)
        nn.init.zeros_(self.Wmean3.weight)

        if self.delta_share_V == False and self.delta_share_U == False:
            self.experts = nn.ModuleList([meanW_deltaUV(config, self.Wmean1, self.Wmean2, self.Wmean3, 
                                                    self.delta_ratio, delta_share_V=False, delta_share_U=False, deepcopy=self.deepcopy)  for _ in range(self.num_experts)])
            
        elif self.delta_share_V == True and self.delta_share_U == False:
            delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))
            self.experts_delta_v1_shared = nn.Linear(self.hidden_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
            self.experts_delta_v2_shared = nn.Linear(self.intermediate_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
            self.experts_delta_v3_shared = nn.Linear(self.hidden_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)

            self.experts = nn.ModuleList([meanW_deltaUV(config, self.Wmean1, self.Wmean2, self.Wmean3, 
                                                    self.delta_ratio, delta_share_V=True, delta_share_U=False, 
                                                    experts_delta_v1_shared=self.experts_delta_v1_shared, 
                                                    experts_delta_v2_shared=self.experts_delta_v2_shared, experts_delta_v3_shared=self.experts_delta_v3_shared, deepcopy=self.deepcopy)  for _ in range(self.num_experts)])
            
        elif self.delta_share_V == True and self.delta_share_U == True:
            delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))
            self.experts_delta_u1_shared = nn.Linear(delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.experts_delta_v1_shared = nn.Linear(self.hidden_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
            self.experts_delta_u2_shared = nn.Linear(delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
            self.experts_delta_v2_shared = nn.Linear(self.intermediate_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)
            self.experts_delta_u3_shared = nn.Linear(delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.experts_delta_v3_shared = nn.Linear(self.hidden_dim, delta_low_rank, bias=False, dtype=torch.bfloat16)

            self.experts = nn.ModuleList([meanW_deltaUV(config, self.Wmean1, self.Wmean2, self.Wmean3, 
                                                    self.delta_ratio, delta_share_V=True, delta_share_U=True, experts_delta_u1_shared=self.experts_delta_u1_shared, experts_delta_v1_shared=self.experts_delta_v1_shared, 
                                                    experts_delta_u2_shared=self.experts_delta_u2_shared, experts_delta_v2_shared=self.experts_delta_v2_shared, 
                                                    experts_delta_u3_shared=self.experts_delta_u3_shared, experts_delta_v3_shared=self.experts_delta_v3_shared, deepcopy=self.deepcopy)  for _ in range(self.num_experts)])

        self.jitter_noise = config.router_jitter_noise

        self.expert_frequency = [0] * self.num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.deepcopy:
            for i in range(self.num_experts):
                self.experts[i].update_Wmean(self.Wmean1, self.Wmean2, self.Wmean3)
                self.experts[i].update_delta_v(self.experts_delta_v1_shared, self.experts_delta_v2_shared, self.experts_delta_v3_shared)
            
            # del self.experts_delta_v1_shared, self.experts_delta_v2_shared, self.experts_delta_v3_shared
            # del self.Wmean1, self.Wmean2, self.Wmean3
        
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() > 0:
                self.expert_frequency[expert_idx] += 1
                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def get_expert_frequencies(self):
        """返回每个专家的激活频率"""
        return self.expert_frequency

    def reset_expert_frequencies(self):
        """重置每个专家的激活频率为零"""
        self.expert_frequency = [0] * self.num_experts

    @staticmethod
    @torch.no_grad()
    def svd_delta(W, ratio=1, svd_scale=None, rank=None, absorb_u=False, absorb_v=False, scale_type='svdllm'):
        if rank is None:
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        else:
            num_s_after_trunc = rank
        if svd_scale is None:
            U, S, VT = torch.linalg.svd(W.float(), full_matrices=False)
            del W
            truc_s = S[:num_s_after_trunc]
            del S
            truc_u = U[:, :num_s_after_trunc]
            del U
            truc_v = VT[:num_s_after_trunc, :]
            del VT
            truc_sigma = torch.diag(truc_s)
            del truc_s
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma)
            svd_v = torch.matmul(sqrtSigma, truc_v)
        else:
            if scale_type == 'svdllm':
                W_scale = torch.matmul(W, svd_scale.bfloat16().to(W.device))
                U, S, VT = torch.linalg.svd(W_scale.float(), full_matrices=False)
                del W_scale
                truc_s = S[:num_s_after_trunc]
                del S
                truc_u = U[:, :num_s_after_trunc]
                del U
                truc_v = torch.matmul(VT[:num_s_after_trunc, :], cal_scale_inv(svd_scale).to(W.device))
                del VT
                truc_sigma = torch.diag(truc_s)
                del truc_s
                if absorb_u:
                    svd_u = torch.matmul(truc_u, truc_sigma)
                    svd_v = truc_v
                elif absorb_v:
                    svd_u = truc_u
                    svd_v = torch.matmul(truc_sigma, truc_v)
                else:
                    sqrtSigma = torch.sqrt(truc_sigma)
                    svd_u = torch.matmul(truc_u, sqrtSigma)
                    svd_v = torch.matmul(sqrtSigma, truc_v)
            elif scale_type == 'asvd':
                alpha = 0.5
                svd_scale *= alpha
                svd_scale += 1e-6
                W_scale = W * svd_scale.to(W.device).view(1, -1)
                U, S, VT = torch.linalg.svd(W_scale.float(), full_matrices=False)
                del W_scale
                truc_s = S[:num_s_after_trunc]
                del S
                truc_u = U[:, :num_s_after_trunc]
                del U
                VT= VT / svd_scale.to(W.device).view(-1, 1)
                truc_v = VT[:num_s_after_trunc, :]
                del VT
                truc_sigma = torch.diag(truc_s)
                del truc_s
                sqrtSigma = torch.sqrt(truc_sigma)
                svd_u = torch.matmul(truc_u, sqrtSigma)
                svd_v = torch.matmul(sqrtSigma, truc_v)
        return svd_u.to(torch.bfloat16), svd_v.to(torch.bfloat16)


    @staticmethod
    @torch.no_grad()
    def svd_delta_reslut(W, ratio=1, svd_scale=None):
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        if svd_scale is None:
            U, S, VT = torch.linalg.svd(W.float(), full_matrices=False)
            del W
            truc_s = S[:num_s_after_trunc]
            del S
            truc_u = U[:, :num_s_after_trunc]
            del U
            truc_v = VT[:num_s_after_trunc, :]
            del VT
            truc_sigma = torch.diag(truc_s)
            del truc_s
            result = truc_u @ truc_sigma @ truc_v
        else:
            W_scale = torch.matmul(W, svd_scale.bfloat16())
            U, S, VT = torch.linalg.svd(W_scale.float(), full_matrices=False)
            del W_scale
            truc_s = S[:num_s_after_trunc]
            del S
            truc_u = U[:, :num_s_after_trunc]
            del U
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], cal_scale_inv(svd_scale))
            del VT
            truc_sigma = torch.diag(truc_s)
            del truc_s
            result = truc_u @ truc_sigma @ truc_v
        return result.to(torch.bfloat16)


    @torch.no_grad()
    def merge_experts(self, module, svd_scale = None, hessian = None, scale_type='svdllm', preprocess_method = None):
        self.gate.weight.data = module.gate.weight.data




        if preprocess_method == "permutation-weight":
            import numpy as np
            domain_idx = np.argmax(self.expert_freq)

            from scipy.optimize import linear_sum_assignment
            @torch.no_grad()
            def compute_switch_permutation_by_weight_matching(reference_expert,target_expert) -> torch.Tensor:
                reference_w1 = reference_expert.w1.weight
                reference_w2 = reference_expert.w2.weight
                reference_w3 = reference_expert.w3.weight

                target_w1 = target_expert.w1.weight
                target_w2 = target_expert.w2.weight
                target_w3 = target_expert.w3.weight

                lsa_cost_matrix = torch.mm(reference_w1.data, target_w1.data.t())
                lsa_cost_matrix += torch.mm(reference_w2.data.t(), target_w2.data)
                lsa_cost_matrix += torch.mm(reference_w3.data, target_w3.data.t())

                _, perm = linear_sum_assignment(lsa_cost_matrix.cpu().float().numpy(), maximize=True)
                return torch.from_numpy(perm).to(lsa_cost_matrix.device)

            @torch.no_grad()
            def permute_switch_mlp_dense_expert_(expert, perm):
                expert.w1.weight.data = expert.w1.weight.data[perm, :]
                expert.w2.weight.data = expert.w2.weight.data[:, perm]
                expert.w3.weight.data = expert.w3.weight.data[perm, :]
                return expert

            for i in range(len(module.experts)):
                if i == domain_idx:
                    continue
                perm = compute_switch_permutation_by_weight_matching(
                    module.experts[domain_idx],
                    module.experts[i],
                )
                module.experts[i] = permute_switch_mlp_dense_expert_(
                    module.experts[i], perm
                )

        if self.merge_method == "freq":
            self.Wmean1.weight.data = sum([module.experts[i].w1.weight * self.expert_freq[i] for i in range(self.num_experts)]) / sum(self.expert_freq)
            self.Wmean2.weight.data = sum([module.experts[i].w2.weight * self.expert_freq[i] for i in range(self.num_experts)]) / sum(self.expert_freq)
            self.Wmean3.weight.data = sum([module.experts[i].w3.weight * self.expert_freq[i] for i in range(self.num_experts)]) / sum(self.expert_freq)

        elif self.merge_method == "freq_DARE":
            def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float=0.9, use_rescale: bool=False, mask_strategy: str="magnitude"):
                """
                mask the input with mask rate
                :param input_tensor: Tensor, input tensor
                :param mask_rate: float, mask rate
                :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
                :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
                :return:
                """
                if mask_rate == 0.0 or mask_rate == 1.0:
                    return input_tensor
                assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
                if mask_strategy == "random":
                    mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
                    masked_input_tensor = input_tensor * (1 - mask)
                else:
                    assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
                    original_shape = input_tensor.shape
                    input_tensor = input_tensor.flatten()
                    num_mask_params = int(len(input_tensor) * mask_rate)
                    # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
                    kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
                    # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
                    mask = input_tensor.abs() <= kth_values
                    masked_input_tensor = input_tensor * (~mask)
                    masked_input_tensor = masked_input_tensor.reshape(original_shape)
                if use_rescale and mask_rate != 1.0:
                    masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
                return masked_input_tensor
            
            self.Wmean1.weight.data = sum([mask_input_with_mask_rate(module.experts[i].w1.weight, mask_rate=0.3) * self.expert_freq[i] for i in range(self.num_experts)]) / sum(self.expert_freq)
            self.Wmean2.weight.data = sum([mask_input_with_mask_rate(module.experts[i].w2.weight, mask_rate=0.3) * self.expert_freq[i] for i in range(self.num_experts)]) / sum(self.expert_freq)
            self.Wmean3.weight.data = sum([mask_input_with_mask_rate(module.experts[i].w3.weight, mask_rate=0.3) * self.expert_freq[i] for i in range(self.num_experts)]) / sum(self.expert_freq)


        elif self.merge_method == "freq_domain":
            scaling_factor = 0.3
            import numpy as np
            domain_idx = np.argmax(self.expert_freq)
            
            normal_freq = [self.expert_freq[i] for i in range(len(self.expert_freq))]
            normal_freq = [normal_freq[i] / sum(normal_freq) for i in range(len(normal_freq))]

            self.Wmean1.weight.data = module.experts[domain_idx].w1.weight + scaling_factor * sum([(module.experts[i].w1.weight - module.experts[domain_idx].w1.weight) * normal_freq[i] for i in range(self.num_experts) if i != domain_idx])
            self.Wmean2.weight.data = module.experts[domain_idx].w2.weight + scaling_factor * sum([(module.experts[i].w2.weight - module.experts[domain_idx].w2.weight) * normal_freq[i] for i in range(self.num_experts) if i != domain_idx])
            self.Wmean3.weight.data = module.experts[domain_idx].w3.weight + scaling_factor * sum([(module.experts[i].w3.weight - module.experts[domain_idx].w3.weight) * normal_freq[i] for i in range(self.num_experts) if i != domain_idx])


        elif self.merge_method == "freq_domain_DARE":
            scaling_factor = 0.3
            import numpy as np
            domain_idx = np.argmax(self.expert_freq)

            def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float=0.5, use_rescale: bool=False, mask_strategy: str="magnitude"):
                if float(mask_rate) == 0.0 or float(mask_rate) == 1.0:
                    return input_tensor
                assert 0.0 <= float(mask_rate) <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
                if mask_strategy == "random":
                    mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
                    masked_input_tensor = input_tensor * (1 - mask)
                else:
                    assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
                    original_shape = input_tensor.shape
                    input_tensor = input_tensor.flatten()
                    num_mask_params = int(len(input_tensor) * mask_rate)
                    # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
                    kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
                    # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
                    mask = input_tensor.abs() <= kth_values
                    masked_input_tensor = input_tensor * (~mask)
                    masked_input_tensor = masked_input_tensor.reshape(original_shape)
                if use_rescale and mask_rate != 1.0:
                    masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
                return masked_input_tensor

            
            normal_freq = [self.expert_freq[i] for i in range(len(self.expert_freq))]
            normal_freq = [normal_freq[i] / sum(normal_freq) for i in range(len(normal_freq))]

            self.Wmean1.weight.data = module.experts[domain_idx].w1.weight + scaling_factor * sum([mask_input_with_mask_rate(module.experts[i].w1.weight - module.experts[domain_idx].w1.weight) * normal_freq[i] for i in range(self.num_experts) if i != domain_idx])
            self.Wmean2.weight.data = module.experts[domain_idx].w2.weight + scaling_factor * sum([mask_input_with_mask_rate(module.experts[i].w2.weight - module.experts[domain_idx].w2.weight) * normal_freq[i] for i in range(self.num_experts) if i != domain_idx])
            self.Wmean3.weight.data = module.experts[domain_idx].w3.weight + scaling_factor * sum([mask_input_with_mask_rate(module.experts[i].w3.weight - module.experts[domain_idx].w3.weight) * normal_freq[i] for i in range(self.num_experts) if i != domain_idx])

        elif self.merge_method == "mean":
            self.Wmean1.weight.data = sum([module.experts[i].w1.weight for i in range(self.num_experts)]) / self.num_experts
            self.Wmean2.weight.data = sum([module.experts[i].w2.weight for i in range(self.num_experts)]) / self.num_experts
            self.Wmean3.weight.data = sum([module.experts[i].w3.weight for i in range(self.num_experts)]) / self.num_experts
        elif self.merge_method == "gradient_domain":
            original_implementation = False
            enable_domain_scaling = True

            scaling_factor = 1
            scaling_factors_ft_list = [1]*8
            scaling_factor_pt = 1
            delta_0 = 1e-12
            import numpy as np
            domain_idx = np.argmax(self.expert_freq)
            if enable_domain_scaling:
                scaling_factors_ft_list[domain_idx] = 0
                scaling_factor_pt = self.expert_freq[domain_idx]
                for i in range(8):
                    if i != domain_idx:
                        scaling_factors_ft_list[i] = self.expert_freq[i]


            domain_hessian_w1 = hessian[f'block_sparse_moe.experts.{domain_idx}.w1']
            domain_hessian_w2 = hessian[f'block_sparse_moe.experts.{domain_idx}.w2']
            domain_hessian_w3 = hessian[f'block_sparse_moe.experts.{domain_idx}.w3']
            other_hessian_w1 = []
            other_hessian_w2 = []
            other_hessian_w3 = []
            other_idx_w1 = []
            other_idx_w2 = []
            other_idx_w3 = []

            for i in range(8):
                if i != domain_idx:
                    other_idx_w1.append(i)
                    other_idx_w2.append(i)
                    other_idx_w3.append(i)
                    other_hessian_w1.append(hessian[f'block_sparse_moe.experts.{i}.w1'])   
                    other_hessian_w2.append(hessian[f'block_sparse_moe.experts.{i}.w2'])   
                    other_hessian_w3.append(hessian[f'block_sparse_moe.experts.{i}.w3'])   
                if i == domain_idx:
                    other_hessian_w1.append(0)
                    other_hessian_w2.append(0)
                    other_hessian_w3.append(0)
            if original_implementation:
                preconditioner_w1 = 1. / (sum([float(scaling_factor) * float(scaling_factor_ft) * ft_hessian
                        for ft_hessian, scaling_factor_ft in zip(other_hessian_w1, scaling_factors_ft_list)]) \
                + float(scaling_factor_pt) * domain_hessian_w1 \
                + delta_0)
                preconditioner_w2 = 1. / (sum([float(scaling_factor) * float(scaling_factor_ft) * ft_hessian
                            for ft_hessian, scaling_factor_ft in zip(other_hessian_w2, scaling_factors_ft_list)]) \
                + float(scaling_factor_pt) * domain_hessian_w2 \
                + delta_0)
                preconditioner_w3 = 1. / (sum([float(scaling_factor) * float(scaling_factor_ft) * ft_hessian
                            for ft_hessian, scaling_factor_ft in zip(other_hessian_w3, scaling_factors_ft_list)]) \
                + float(scaling_factor_pt) * domain_hessian_w3 \
                + delta_0)

                summed_w1 = sum([  (preconditioner_w1 * (float(scaling_factor_ft) * ft_hessian + delta_0 + float(scaling_factor_pt) * domain_hessian_w1  ))
                                * (module.experts[domain_idx].w1.weight - module.experts[i].w1.weight) 
                                for i, ft_hessian, scaling_factor_ft in zip(other_idx_w1, other_hessian_w1, scaling_factors_ft_list)])
                summed_w2 = sum([  (preconditioner_w2 * (float(scaling_factor_ft) * ft_hessian + delta_0 + float(scaling_factor_pt) * domain_hessian_w2  ))
                                * (module.experts[domain_idx].w2.weight - module.experts[i].w2.weight) 
                                for i, ft_hessian, scaling_factor_ft in zip(other_idx_w2, other_hessian_w2, scaling_factors_ft_list)])
                summed_w3 = sum([  (preconditioner_w3 * (float(scaling_factor_ft) * ft_hessian + delta_0 + float(scaling_factor_pt) * domain_hessian_w3  ))
                                * (module.experts[domain_idx].w3.weight - module.experts[i].w3.weight) 
                                for i, ft_hessian, scaling_factor_ft in zip(other_idx_w3, other_hessian_w3, scaling_factors_ft_list)])

                self.Wmean1.weight.data = module.experts[domain_idx].w1.weight + float(scaling_factor) * summed_w1
                self.Wmean2.weight.data = module.experts[domain_idx].w2.weight + float(scaling_factor) * summed_w2
                self.Wmean3.weight.data = module.experts[domain_idx].w3.weight + float(scaling_factor) * summed_w3

            else:
                hessian_average_w1 = domain_hessian_w1 * scaling_factor_pt + sum([other_hessian_w1[i] * scaling_factors_ft_list[i] for i in range(8) if i != domain_idx]) + delta_0
                hessian_average_w2 = domain_hessian_w2 * scaling_factor_pt + sum([other_hessian_w2[i] * scaling_factors_ft_list[i] for i in range(8) if i != domain_idx]) + delta_0
                hessian_average_w3 = domain_hessian_w3 * scaling_factor_pt + sum([other_hessian_w3[i] * scaling_factors_ft_list[i] for i in range(8) if i != domain_idx]) + delta_0

                self.Wmean1.weight.data = module.experts[domain_idx].w1.weight + scaling_factor * sum([ (other_hessian_w1[i] * scaling_factors_ft_list[i] + domain_hessian_w1 * scaling_factor_pt + delta_0) * (module.experts[i].w1.weight - module.experts[domain_idx].w1.weight) for i in range(8) if i != domain_idx]) / hessian_average_w1
                self.Wmean2.weight.data = module.experts[domain_idx].w2.weight + scaling_factor * sum([ (other_hessian_w2[i] * scaling_factors_ft_list[i] + domain_hessian_w2 * scaling_factor_pt + delta_0) * (module.experts[i].w2.weight - module.experts[domain_idx].w2.weight) for i in range(8) if i != domain_idx]) / hessian_average_w2
                self.Wmean3.weight.data = module.experts[domain_idx].w3.weight + scaling_factor * sum([ (other_hessian_w3[i] * scaling_factors_ft_list[i] + domain_hessian_w3 * scaling_factor_pt + delta_0) * (module.experts[i].w3.weight - module.experts[domain_idx].w3.weight) for i in range(8) if i != domain_idx]) / hessian_average_w3
        elif self.merge_method == "fisher":
            scaling_factor = 1.0
            scaling_factors_ft_list = self.expert_freq

            fisher_scale_w1 = [0]*8
            fisher_scale_w2 = [0]*8
            fisher_scale_w3 = [0]*8

            for j in range(8):
                base_name = f"block_sparse_moe.experts.{j}."
                fisher_scale_w1[j] = hessian[base_name + "w1"].to(module.experts[j].w1.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w2[j] = hessian[base_name + "w2"].to(module.experts[j].w2.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w3[j] = hessian[base_name + "w3"].to(module.experts[j].w3.weight.device) * scaling_factors_ft_list[j]

            fisher_scale_w1 = [fisher_scale_w1[j] / sum(fisher_scale_w1) for j in range(8)]
            fisher_scale_w2 = [fisher_scale_w2[j] / sum(fisher_scale_w2) for j in range(8)]
            fisher_scale_w3 = [fisher_scale_w3[j] / sum(fisher_scale_w3) for j in range(8)]
                
            self.Wmean1.weight.data = sum([module.experts[j].w1.weight * fisher_scale_w1[j] for j in range(8)])
            self.Wmean2.weight.data = sum([module.experts[j].w2.weight * fisher_scale_w2[j] for j in range(8)])
            self.Wmean3.weight.data = sum([module.experts[j].w3.weight * fisher_scale_w3[j] for j in range(8)])

        elif self.merge_method == "fisher_else":
            scaling_factor = 1.0
            scaling_factors_ft_list = self.expert_freq

            fisher_scale_w1 = [0]*8
            fisher_scale_w2 = [0]*8
            fisher_scale_w3 = [0]*8

            for j in range(8):
                base_name = f"block_sparse_moe.experts.{j}."
                fisher_scale_w1[j] = hessian[base_name + "w1"].to(module.experts[j].w1.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w2[j] = hessian[base_name + "w2"].to(module.experts[j].w2.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w3[j] = hessian[base_name + "w3"].to(module.experts[j].w3.weight.device) * scaling_factors_ft_list[j]

            fisher_scale_w1 = [fisher_scale_w1[j] / sum(fisher_scale_w1) for j in range(8)]
            fisher_scale_w2 = [fisher_scale_w2[j] / sum(fisher_scale_w2) for j in range(8)]
            fisher_scale_w3 = [fisher_scale_w3[j] / sum(fisher_scale_w3) for j in range(8)]
                
            self.Wmean1.weight.data = sum([torch.matmul(module.experts[j].w1.weight, fisher_scale_w1[j]) for j in range(8)])
            self.Wmean2.weight.data = sum([torch.matmul(module.experts[j].w2.weight, fisher_scale_w2[j]) for j in range(8)])
            self.Wmean3.weight.data = sum([torch.matmul(module.experts[j].w3.weight, fisher_scale_w3[j]) for j in range(8)])

        elif self.merge_method == "fisher_domain":
            scaling_factor = 1
            scaling_factors_ft_list = self.expert_freq
            import numpy as np
            domain_idx = np.argmax(self.expert_freq)

            fisher_scale_w1 = [1]*8
            fisher_scale_w2 = [1]*8
            fisher_scale_w3 = [1]*8

            for j in range(8):
                base_name = f"block_sparse_moe.experts.{j}."
                fisher_scale_w1[j] = hessian[base_name + "w1"].to(module.experts[j].w1.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w2[j] = hessian[base_name + "w2"].to(module.experts[j].w2.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w3[j] = hessian[base_name + "w3"].to(module.experts[j].w3.weight.device) * scaling_factors_ft_list[j]

            fisher_scale_w1 = [fisher_scale_w1[j] / sum(fisher_scale_w1) for j in range(8)]
            fisher_scale_w2 = [fisher_scale_w2[j] / sum(fisher_scale_w2) for j in range(8)]
            fisher_scale_w3 = [fisher_scale_w3[j] / sum(fisher_scale_w3) for j in range(8)]
                
            self.Wmean1.weight.data = module.experts[domain_idx].w1.weight + scaling_factor * sum([ (fisher_scale_w1[j] * module.experts[j].w1.weight - fisher_scale_w1[domain_idx] * module.experts[domain_idx].w1.weight) for j in range(8) if j != domain_idx])
            self.Wmean2.weight.data = module.experts[domain_idx].w2.weight + scaling_factor * sum([ (fisher_scale_w2[j] * module.experts[j].w2.weight - fisher_scale_w2[domain_idx] * module.experts[domain_idx].w2.weight) for j in range(8) if j != domain_idx])
            self.Wmean3.weight.data = module.experts[domain_idx].w3.weight + scaling_factor * sum([ (fisher_scale_w3[j] * module.experts[j].w3.weight - fisher_scale_w3[domain_idx] * module.experts[domain_idx].w3.weight) for j in range(8) if j != domain_idx])

        elif self.merge_method == "fisher_domain_DARE":
            scaling_factor = 0.5
            scaling_factors_ft_list = [1]*8
            scaling_factors_ft_list = self.expert_freq
            import numpy as np
            domain_idx = np.argmax(self.expert_freq)

            fisher_scale_w1 = [1]*8
            fisher_scale_w2 = [1]*8
            fisher_scale_w3 = [1]*8

            for j in range(8):
                base_name = f"block_sparse_moe.experts.{j}."
                fisher_scale_w1[j] = hessian[base_name + "w1"].to(module.experts[j].w1.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w2[j] = hessian[base_name + "w2"].to(module.experts[j].w2.weight.device) * scaling_factors_ft_list[j]
                fisher_scale_w3[j] = hessian[base_name + "w3"].to(module.experts[j].w3.weight.device) * scaling_factors_ft_list[j]

            fisher_scale_w1 = [fisher_scale_w1[j] / sum(fisher_scale_w1) for j in range(8)]
            fisher_scale_w2 = [fisher_scale_w2[j] / sum(fisher_scale_w2) for j in range(8)]
            fisher_scale_w3 = [fisher_scale_w3[j] / sum(fisher_scale_w3) for j in range(8)]


            def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float=0.9, use_rescale: bool=False, mask_strategy: str="magnitude"):
                """
                mask the input with mask rate
                :param input_tensor: Tensor, input tensor
                :param mask_rate: float, mask rate
                :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
                :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
                :return:
                """
                if mask_rate == 0.0 or mask_rate == 1.0:
                    return input_tensor
                assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
                if mask_strategy == "random":
                    mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
                    masked_input_tensor = input_tensor * (1 - mask)
                else:
                    assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
                    original_shape = input_tensor.shape
                    input_tensor = input_tensor.flatten()
                    num_mask_params = int(len(input_tensor) * mask_rate)
                    # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
                    kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
                    # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
                    mask = input_tensor.abs() <= kth_values
                    masked_input_tensor = input_tensor * (~mask)
                    masked_input_tensor = masked_input_tensor.reshape(original_shape)
                if use_rescale and mask_rate != 1.0:
                    masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
                return masked_input_tensor
                
            self.Wmean1.weight.data = module.experts[domain_idx].w1.weight + scaling_factor * sum([ mask_input_with_mask_rate(module.experts[j].w1.weight * fisher_scale_w1[j] - module.experts[domain_idx].w1.weight * fisher_scale_w1[domain_idx]) for j in range(8) if j != domain_idx])
            self.Wmean2.weight.data = module.experts[domain_idx].w2.weight + scaling_factor * sum([ mask_input_with_mask_rate(module.experts[j].w2.weight * fisher_scale_w2[j] - module.experts[domain_idx].w2.weight * fisher_scale_w2[domain_idx]) for j in range(8) if j != domain_idx])
            self.Wmean3.weight.data = module.experts[domain_idx].w3.weight + scaling_factor * sum([ mask_input_with_mask_rate(module.experts[j].w3.weight * fisher_scale_w3[j] - module.experts[domain_idx].w3.weight * fisher_scale_w3[domain_idx]) for j in range(8) if j != domain_idx])
        

        elif self.merge_method == "random":
            torch.manual_seed(111)
            scale_alpha_1 = torch.rand(8, device=module.experts[0].w1.weight.device)
            torch.manual_seed(222)
            scale_alpha_2 = torch.rand(8, device=module.experts[0].w1.weight.device)
            torch.manual_seed(333)
            scale_alpha_3 = torch.rand(8, device=module.experts[0].w1.weight.device)
            scale_alpha_1 = scale_alpha_1 / scale_alpha_1.sum()  # Normalize to sum to 1
            scale_alpha_2 = scale_alpha_2 / scale_alpha_2.sum()  # Normalize to sum to 1
            scale_alpha_3 = scale_alpha_3 / scale_alpha_3.sum()  # Normalize to sum to 1

            self.Wmean1.weight.data = sum([module.experts[i].w1.weight * scale_alpha_1[i] for i in range(8)])
            self.Wmean2.weight.data = sum([module.experts[i].w2.weight * scale_alpha_2[i] for i in range(8)])
            self.Wmean3.weight.data = sum([module.experts[i].w3.weight * scale_alpha_3[i] for i in range(8)])

        elif self.merge_method == "weighted":
            w1_weight_list = torch.stack([module.experts[i].w1.weight * module.experts[i].w1.weight for i in range(8)], dim=0)
            w2_weight_list = torch.stack([module.experts[i].w2.weight * module.experts[i].w2.weight for i in range(8)], dim=0)
            w3_weight_list = torch.stack([module.experts[i].w3.weight * module.experts[i].w3.weight for i in range(8)], dim=0)

            self.Wmean1.weight.data = sum(torch.sum(w1_weight_list, dim=0)) / sum([module.experts[i].w1.weight for i in range(8)])
            self.Wmean2.weight.data = sum(torch.sum(w2_weight_list, dim=0)) / sum([module.experts[i].w2.weight for i in range(8)])
            self.Wmean3.weight.data = sum(torch.sum(w3_weight_list, dim=0)) / sum([module.experts[i].w3.weight for i in range(8)])

        elif self.merge_method == "regmean":
            grams = hessian
            def reduce_non_diag(cov_mat, a):
                diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
                non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
                weight = diag_weight + non_diag_weight
                ret = cov_mat * weight
                return ret
            
            for _, gram in grams.items():
                gram = reduce_non_diag(gram, a=0.5)

            sum_w1 = 0
            sum_w2 = 0
            sum_w3 = 0
            sum_gram_w1 = 0
            sum_gram_w2 = 0
            sum_gram_w3 = 0
            for j in range(8):
                base_name = f"block_sparse_moe.experts.{j}."
                sum_w1 += torch.matmul(grams[base_name + "w1"], module.experts[j].w1.weight.transpose(0,1))
                sum_w2 += torch.matmul(grams[base_name + "w2"], module.experts[j].w2.weight.transpose(0,1))
                sum_w3 += torch.matmul(grams[base_name + "w3"], module.experts[j].w3.weight.transpose(0,1))

                sum_gram_w1 += grams[base_name + "w1"]
                sum_gram_w2 += grams[base_name + "w2"]
                sum_gram_w3 += grams[base_name + "w3"]

            self.Wmean1.weight.data = torch.matmul(torch.inverse(sum_gram_w1.to(torch.float32)).to(torch.bfloat16), sum_w1).transpose(0,1)
            self.Wmean2.weight.data = torch.matmul(torch.inverse(sum_gram_w2.to(torch.float32)).to(torch.bfloat16), sum_w2).transpose(0,1)
            self.Wmean3.weight.data = torch.matmul(torch.inverse(sum_gram_w3.to(torch.float32)).to(torch.bfloat16), sum_w3).transpose(0,1)
        
        elif self.merge_method == "ties":
            lamda = 1
            import numpy as np
            domain_idx = np.argmax(self.expert_freq)

            def to_vector(tensor):
                return tensor.flatten()

            def topk_values_mask(M, K=0.7, return_mask=False):
                if K > 1:
                    K /= 100

                original_shape = M.shape
                if M.dim() == 1:
                    M = M.unsqueeze(0)

                n, d = M.shape
                k = int(d * K)
                k = d - k  # Keep top k elements instead of bottom k elements

                # Find the k-th smallest element by magnitude for each row
                kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
                # Create a mask tensor with True for the top k elements in each row
                mask = M.abs() >= kth_values
                final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

                if return_mask:
                    return M * final_mask, final_mask.float().mean(dim=1), final_mask
                return M * final_mask
            def resolve_zero_signs(sign_to_mult, method="majority"):
                majority_sign = torch.sign(sign_to_mult.sum())

                if method == "majority":
                    sign_to_mult[sign_to_mult == 0] = majority_sign
                elif method == "minority":
                    sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
                return sign_to_mult

            def resolve_sign(Tensor):
                sign_to_mult = torch.sign(Tensor.sum(dim=0))
                sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
                return sign_to_mult

            def disjoint_merge(Tensor, merge_func, sign_to_mult):

                merge_func = merge_func.split("-")[-1]

                # If sign is provided then we select the corresponding entries and aggregate.
                if sign_to_mult is not None:
                    rows_to_keep = torch.where(
                        sign_to_mult.unsqueeze(0) > 0, Tensor > 0, Tensor < 0
                    )
                    selected_entries = Tensor * rows_to_keep
                # Else we select all non-zero entries and aggregate.
                else:
                    rows_to_keep = Tensor != 0
                    selected_entries = Tensor * rows_to_keep

                if merge_func == "mean":
                    non_zero_counts = (selected_entries != 0).sum(dim=0).float()
                    disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
                        non_zero_counts, min=1
                    )
                elif merge_func == "sum":
                    disjoint_aggs = torch.sum(selected_entries, dim=0)
                elif merge_func == "max":
                    disjoint_aggs = selected_entries.abs().max(dim=0)[0]
                    disjoint_aggs *= sign_to_mult
                else:
                    raise ValueError(f"Merge method {merge_func} is not defined.")

                return disjoint_aggs

            delta_w1 = torch.zeros(7, module.experts[0].w1.weight.shape[0] * module.experts[0].w1.weight.shape[1], device=module.experts[0].w1.weight.device)
            delta_w2 = torch.zeros(7, module.experts[0].w2.weight.shape[0] * module.experts[0].w2.weight.shape[1], device=module.experts[0].w2.weight.device)
            delta_w3 = torch.zeros(7, module.experts[0].w3.weight.shape[0] * module.experts[0].w3.weight.shape[1], device=module.experts[0].w3.weight.device)
            idx = 0
            for j in range(8):
                if j == domain_idx:
                    continue
                delta_w1[idx] = to_vector(module.experts[j].w1.weight - module.experts[domain_idx].w1.weight)
                delta_w2[idx] = to_vector(module.experts[j].w2.weight - module.experts[domain_idx].w2.weight)
                delta_w3[idx] = to_vector(module.experts[j].w3.weight - module.experts[domain_idx].w3.weight)
                idx += 1

            update_w1 = topk_values_mask(delta_w1, K=20)
            update_w2 = topk_values_mask(delta_w2, K=20)
            update_w3 = topk_values_mask(delta_w3, K=20)

            final_sign_w1 = resolve_sign(update_w1)
            final_sign_w2 = resolve_sign(update_w2)
            final_sign_w3 = resolve_sign(update_w3)

            merged_w1 = disjoint_merge(update_w1, "max", final_sign_w1)
            merged_w2 = disjoint_merge(update_w2, "max", final_sign_w2)
            merged_w3 = disjoint_merge(update_w3, "max", final_sign_w3)

            self.Wmean1.weight.data = (module.experts[domain_idx].w1.weight + lamda * merged_w1.reshape(module.experts[domain_idx].w1.weight.shape)).to(torch.bfloat16)
            self.Wmean2.weight.data = (module.experts[domain_idx].w2.weight + lamda * merged_w2.reshape(module.experts[domain_idx].w2.weight.shape)).to(torch.bfloat16)
            self.Wmean3.weight.data = (module.experts[domain_idx].w3.weight + lamda * merged_w3.reshape(module.experts[domain_idx].w3.weight.shape)).to(torch.bfloat16)            
        
        else:
            raise ValueError(f"wrong merge method {self.merge_method}!")



        scale_w1_mean = None
        scale_w2_mean = None
        scale_w3_mean = None
        total_freq = 0
        if svd_scale is not None:
            for j in range(self.num_experts):
                base_name = f"block_sparse_moe.experts.{j}."
                freq = self.expert_freq[j]  
                total_freq += freq
                if scale_w1_mean is None:
                    scale_w1_mean = svd_scale[base_name + "w1"] * freq
                else:
                    scale_w1_mean += svd_scale[base_name + "w1"] * freq
                if scale_w2_mean is None:
                    scale_w2_mean = svd_scale[base_name + "w2"] * freq
                else:
                    scale_w2_mean += svd_scale[base_name + "w2"] * freq
                if scale_w3_mean is None:
                    scale_w3_mean = svd_scale[base_name + "w3"] * freq
                else:
                    scale_w3_mean += svd_scale[base_name + "w3"] * freq
            scale_w1_mean /= total_freq
            scale_w2_mean /= total_freq
            scale_w3_mean /= total_freq

        if self.delta_share_V == True and self.delta_share_U == False:
            delta_w1 = []
            delta_w2 = []
            delta_w3 = []

            for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):
                
                delta_w1.append(module.experts[j].w1.weight - self.Wmean1.weight)
                delta_w2.append(module.experts[j].w2.weight - self.Wmean2.weight)
                delta_w3.append(module.experts[j].w3.weight - self.Wmean3.weight)

            delta_w1 = torch.stack(delta_w1, dim=0).reshape(-1, delta_w1[0].shape[1])
            delta_w2 = torch.stack(delta_w2, dim=0).reshape(-1, delta_w2[0].shape[1])
            delta_w3 = torch.stack(delta_w3, dim=0).reshape(-1, delta_w3[0].shape[1])

            if svd_scale is None:
                delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank)
                delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank)
                delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank)
            else:
                delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank, svd_scale=scale_w1_mean, scale_type=scale_type)
                delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank, svd_scale=scale_w2_mean, scale_type=scale_type)
                delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank, svd_scale=scale_w3_mean, scale_type=scale_type)

            shared_v1 = nn.Parameter(shared_v1)
            shared_v2 = nn.Parameter(shared_v2)
            shared_v3 = nn.Parameter(shared_v3)

            del delta_w1, delta_w2, delta_w3

            self.experts_delta_v1_shared.weight = shared_v1
            self.experts_delta_v2_shared.weight = shared_v2
            self.experts_delta_v3_shared.weight = shared_v3

            for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):

                self.experts[j].delta_u1.weight.data = delta_u1[j * self.experts[j].delta_u1.weight.shape[0]:(j + 1) * self.experts[j].delta_u1.weight.shape[0], :]
                # self.experts[j].delta_v1.weight = shared_v1
                self.experts[j].delta_u2.weight.data = delta_u2[j * self.experts[j].delta_u2.weight.shape[0]:(j + 1) * self.experts[j].delta_u2.weight.shape[0], :]
                # self.experts[j].delta_v2.weight = shared_v2
                self.experts[j].delta_u3.weight.data = delta_u3[j * self.experts[j].delta_u3.weight.shape[0]:(j + 1) * self.experts[j].delta_u3.weight.shape[0], :]
                # self.experts[j].delta_v3.weight = shared_v3

        # if self.delta_share_V == False and self.delta_share_U == True:
        #     delta_w1 = []
        #     delta_w2 = []
        #     delta_w3 = []

            # for j in range(self.num_experts):                
            #     delta_w1.append(module.experts[j].w1.weight - self.Wmean1.weight)
            #     delta_w2.append(module.experts[j].w2.weight - self.Wmean2.weight)
            #     delta_w3.append(module.experts[j].w3.weight - self.Wmean3.weight)

        #     delta_w1 = torch.stack(delta_w1, dim=1).reshape(delta_w1[0].shape[0], -1)
        #     delta_w2 = torch.stack(delta_w2, dim=1).reshape(delta_w2[0].shape[0], -1) 
        #     delta_w3 = torch.stack(delta_w3, dim=1).reshape(delta_w3[0].shape[0], -1)

        #     if svd_scale is None:
        #         shared_u1, delta_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank)
        #         shared_u2, delta_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank)
        #         shared_u3, delta_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank)
        #     else:
        #         shared_u1, delta_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank, svd_scale=scale_w1_mean)
        #         shared_u2, delta_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank, svd_scale=scale_w2_mean)
        #         shared_u3, delta_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank, svd_scale=scale_w3_mean)

        #     shared_u1 = nn.Parameter(shared_u1)
        #     shared_u2 = nn.Parameter(shared_u2)
        #     shared_u3 = nn.Parameter(shared_u3)

        #     for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):

        #         self.experts[j].delta_u1.weight = shared_u1
        #         self.experts[j].delta_v1.weight.data = delta_v1[:, j * self.experts[j].delta_v1.weight.shape[1]:(j + 1) * self.experts[j].delta_v1.weight.shape[1]]
        #         self.experts[j].delta_u2.weight = shared_u2
        #         self.experts[j].delta_v2.weight.data = delta_v2[:, j * self.experts[j].delta_v2.weight.shape[1]:(j + 1) * self.experts[j].delta_v2.weight.shape[1]]
        #         self.experts[j].delta_u3.weight = shared_u3
        #         self.experts[j].delta_v3.weight.data = delta_v3[:, j * self.experts[j].delta_v3.weight.shape[1]:(j + 1) * self.experts[j].delta_v3.weight.shape[1]]



        if self.delta_share_V == True and self.delta_share_U == True:
            delta_w1 = []
            delta_w2 = []
            delta_w3 = []

            for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):               
                delta_w1.append(module.experts[j].w1.weight - self.Wmean1.weight)
                delta_w2.append(module.experts[j].w2.weight - self.Wmean2.weight)
                delta_w3.append(module.experts[j].w3.weight - self.Wmean3.weight)

            delta_w1 = torch.stack(delta_w1, dim=0).reshape(-1, delta_w1[0].shape[1])
            delta_w2 = torch.stack(delta_w2, dim=0).reshape(-1, delta_w2[0].shape[1])
            delta_w3 = torch.stack(delta_w3, dim=0).reshape(-1, delta_w3[0].shape[1])

            if svd_scale is None:
                delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank)
                delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank)
                delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank)
            else:
                delta_u1, shared_v1 = self.svd_delta(delta_w1, rank=self.experts[0].delta_low_rank, svd_scale=scale_w1_mean, scale_type=scale_type)
                delta_u2, shared_v2 = self.svd_delta(delta_w2, rank=self.experts[0].delta_low_rank, svd_scale=scale_w2_mean, scale_type=scale_type)
                delta_u3, shared_v3 = self.svd_delta(delta_w3, rank=self.experts[0].delta_low_rank, svd_scale=scale_w3_mean, scale_type=scale_type)

            shared_v1 = nn.Parameter(shared_v1)
            shared_v2 = nn.Parameter(shared_v2)
            shared_v3 = nn.Parameter(shared_v3)

            # j = 0
            # shared_u1  = delta_u1[j * self.experts[j].delta_u1.weight.shape[0]:(j + 1) * self.experts[j].delta_u1.weight.shape[0], :]
            # shared_u2  = delta_u2[j * self.experts[j].delta_u2.weight.shape[0]:(j + 1) * self.experts[j].delta_u2.weight.shape[0], :]
            # shared_u3  = delta_u3[j * self.experts[j].delta_u3.weight.shape[0]:(j + 1) * self.experts[j].delta_u3.weight.shape[0], :]
            shared_u1 = None
            shared_u2 = None
            shared_u3 = None
            total_freq = 0
            for j in range(self.num_experts):
                freq = self.expert_freq[j]
                total_freq += freq
                if shared_u1 is None:
                    shared_u1 = delta_u1[j * self.experts[j].delta_u1.weight.shape[0]:(j + 1) * self.experts[j].delta_u1.weight.shape[0], :] * freq 
                else:
                    shared_u1 += delta_u1[j * self.experts[j].delta_u1.weight.shape[0]:(j + 1) * self.experts[j].delta_u1.weight.shape[0], :] * freq
                if shared_u2 is None:
                    shared_u2 = delta_u2[j * self.experts[j].delta_u2.weight.shape[0]:(j + 1) * self.experts[j].delta_u2.weight.shape[0], :] * freq
                else:
                    shared_u2 += delta_u2[j * self.experts[j].delta_u2.weight.shape[0]:(j + 1) * self.experts[j].delta_u2.weight.shape[0], :] * freq
                if shared_u3 is None:
                    shared_u3 = delta_u3[j * self.experts[j].delta_u3.weight.shape[0]:(j + 1) * self.experts[j].delta_u3.weight.shape[0], :] * freq
                else:
                    shared_u3 += delta_u3[j * self.experts[j].delta_u3.weight.shape[0]:(j + 1) * self.experts[j].delta_u3.weight.shape[0], :] * freq
            shared_u1 /= total_freq
            shared_u2 /= total_freq
            shared_u3 /= total_freq

            shared_u1 = nn.Parameter(shared_u1)
            shared_u2 = nn.Parameter(shared_u2)
            shared_u3 = nn.Parameter(shared_u3)

            self.experts_delta_u1_shared.weight = shared_u1
            self.experts_delta_v1_shared.weight = shared_v1
            self.experts_delta_u2_shared.weight = shared_u2
            self.experts_delta_v2_shared.weight = shared_v2
            self.experts_delta_u3_shared.weight = shared_u3
            self.experts_delta_v3_shared.weight = shared_v3

            # for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):

            #     self.experts[j].delta_u1.weight = shared_u1
            #     self.experts[j].delta_v1.weight = shared_v1
            #     self.experts[j].delta_u2.weight = shared_u2
            #     self.experts[j].delta_v2.weight = shared_v2
            #     self.experts[j].delta_u3.weight = shared_u3
            #     self.experts[j].delta_v3.weight = shared_v3

        if self.delta_share_V == False and self.delta_share_U == False:
            for j in tqdm(range(self.num_experts), desc="Merging experts", leave=False):
                delta_w1 = (module.experts[j].w1.weight - self.Wmean1.weight)
                delta_w2 = (module.experts[j].w2.weight - self.Wmean2.weight)
                delta_w3 = (module.experts[j].w3.weight - self.Wmean3.weight)

                if svd_scale is not None:
                    base_name = f"block_sparse_moe.experts.{j}."
                    self.experts[j].delta_u1.weight.data, self.experts[j].delta_v1.weight.data = self.svd_delta(delta_w1, ratio=self.delta_ratio, svd_scale=svd_scale[base_name + "w1"], scale_type=scale_type)
                    self.experts[j].delta_u2.weight.data, self.experts[j].delta_v2.weight.data = self.svd_delta(delta_w2, ratio=self.delta_ratio, svd_scale=svd_scale[base_name + "w2"], scale_type=scale_type)
                    self.experts[j].delta_u3.weight.data, self.experts[j].delta_v3.weight.data = self.svd_delta(delta_w3, ratio=self.delta_ratio, svd_scale=svd_scale[base_name + "w3"], scale_type=scale_type)
                else:
                    self.experts[j].delta_u1.weight.data, self.experts[j].delta_v1.weight.data = self.svd_delta(delta_w1, ratio=self.delta_ratio)
                    self.experts[j].delta_u2.weight.data, self.experts[j].delta_v2.weight.data = self.svd_delta(delta_w2, ratio=self.delta_ratio)
                    self.experts[j].delta_u3.weight.data, self.experts[j].delta_v3.weight.data = self.svd_delta(delta_w3, ratio=self.delta_ratio)
        
        del svd_scale




class delta_weight_linear(nn.Module):
    def __init__(self, config: MixtralConfig, ratio=1):
        super().__init__()
        self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = torch.bfloat16

        self.w1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)
        self.w2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=self.dtype)
        self.w3 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)

        self.act_fn = ACT2FN[config.hidden_act]

        self.delta_w1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.delta_w2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.delta_w3 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, hidden_states):
        up = self.w3(hidden_states) + self.delta_w3(hidden_states)
        gate = self.w1(hidden_states) + self.delta_w1(hidden_states)
        return self.w2(self.act_fn(gate) * up) + self.delta_w2(self.act_fn(gate) * up)


class share_expert_with_delta_weight(nn.Module):
    def __init__(self, config: MixtralConfig, share_ratio=1, delta_ratio=1):
        super().__init__()
        self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = torch.bfloat16

        self.w1 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)
        self.w2 = nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False, dtype=self.dtype)
        self.w3 = nn.Linear(self.hidden_dim, self.intermediate_dim, bias=False, dtype=self.dtype)

        self.act_fn = ACT2FN[config.hidden_act]

        self.delta_ratio = delta_ratio
        self.delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))

        self.u1 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v1 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        self.u2 = nn.Linear(self.delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v2 = nn.Linear(self.intermediate_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        self.u3 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v3 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        # v is right, u is left

    def forward(self, hidden_states):
        up = self.w3(hidden_states) + self.u3(self.v3(hidden_states))
        gate = self.w1(hidden_states) + self.u1(self.v1(hidden_states))
        return self.w2(self.act_fn(gate) * up) + self.u2(self.v2(self.act_fn(gate) * up))
    

class share_svd_expert_with_delta_weight(nn.Module):
    def __init__(self, config: MixtralConfig, share_ratio=1, delta_ratio=1):
        super().__init__()
        self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = torch.bfloat16

        self.share_ratio = share_ratio
        self.share_low_rank = int(self.intermediate_dim * self.hidden_dim * self.share_ratio / (self.intermediate_dim + self.hidden_dim))


        self.w1_u = nn.Linear(self.share_low_rank, self.intermediate_dim, bias=False, dtype=self.dtype)
        self.w1_v = nn.Linear(self.hidden_dim, self.share_low_rank, bias=False, dtype=self.dtype)
        
        self.w2_u = nn.Linear(self.share_low_rank, self.hidden_dim, bias=False, dtype=self.dtype)
        self.w2_v = nn.Linear(self.intermediate_dim, self.share_low_rank, bias=False, dtype=self.dtype)
        
        self.w3_u = nn.Linear(self.share_low_rank, self.intermediate_dim, bias=False, dtype=self.dtype)
        self.w3_v = nn.Linear(self.hidden_dim, self.share_low_rank, bias=False, dtype=self.dtype)



        self.act_fn = ACT2FN[config.hidden_act]

        self.delta_ratio = delta_ratio
        self.delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))

        self.u1 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v1 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        self.u2 = nn.Linear(self.delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
        self.v2 = nn.Linear(self.intermediate_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        self.u3 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
        self.v3 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        # v is right, u is left

    def forward(self, hidden_states):
        up = self.w3_u(self.w3_v(hidden_states)) + self.u3(self.v3(hidden_states))
        gate = self.w1_u(self.w1_v(hidden_states)) + self.u1(self.v1(hidden_states))
        return self.w2_u(self.w2_v(self.act_fn(gate) * up)) + self.u2(self.v2(self.act_fn(gate) * up))
    


class meanW_deltaUV(nn.Module):
    def __init__(self, config: MixtralConfig, Wmean1, Wmean2, Wmean3, delta_ratio=1, delta_share_V=False, delta_share_U=False, 
                 experts_delta_u1_shared=None, experts_delta_v1_shared=None, 
                 experts_delta_u2_shared=None, experts_delta_v2_shared=None, 
                 experts_delta_u3_shared=None, experts_delta_v3_shared=None, deepcopy=False):
        super().__init__()
        self.intermediate_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.dtype = torch.bfloat16

        if not deepcopy:
            self.Wmean1 = Wmean1
            self.Wmean2 = Wmean2
            self.Wmean3 = Wmean3
        else:
            self.Wmean1 = nn.Linear(Wmean1.in_features, Wmean1.out_features, bias=Wmean1.bias is not None, dtype=self.dtype)
            self.Wmean1.weight.data = Wmean1.weight.data.clone()
            if Wmean1.bias is not None:
                self.Wmean1.bias.data = Wmean1.bias.data.clone()
            self.Wmean2 = nn.Linear(Wmean2.in_features, Wmean2.out_features, bias=Wmean2.bias is not None, dtype=self.dtype)
            self.Wmean2.weight.data = Wmean2.weight.data.clone()
            if Wmean2.bias is not None:
                self.Wmean2.bias.data = Wmean2.bias.data.clone()
            self.Wmean3 = nn.Linear(Wmean3.in_features, Wmean3.out_features, bias=Wmean3.bias is not None, dtype=self.dtype)
            self.Wmean3.weight.data = Wmean3.weight.data.clone()
            if Wmean3.bias is not None:
                self.Wmean3.bias.data = Wmean3.bias.data.clone()
            
            delta_share_U = False
            delta_share_V = False
            
        self.delta_ratio = delta_ratio

        self.act_fn = ACT2FN[config.hidden_act]

        if delta_share_V == False and delta_share_U == False:
            self.delta_ratio = delta_ratio
            self.delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))
            self.delta_u1 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v1 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)
            self.delta_u2 = nn.Linear(self.delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v2 = nn.Linear(self.intermediate_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)
            self.delta_u3 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v3 = nn.Linear(self.hidden_dim, self.delta_low_rank, bias=False, dtype=torch.bfloat16)

        elif delta_share_V == True and delta_share_U == False:
            self.delta_ratio = delta_ratio
            self.delta_low_rank = int(self.intermediate_dim * self.hidden_dim * self.delta_ratio / (self.intermediate_dim + self.hidden_dim))
            self.delta_u1 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v1 = experts_delta_v1_shared
            self.delta_u2 = nn.Linear(self.delta_low_rank, self.hidden_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v2 = experts_delta_v2_shared
            self.delta_u3 = nn.Linear(self.delta_low_rank, self.intermediate_dim, bias=False, dtype=torch.bfloat16)
            self.delta_v3 = experts_delta_v3_shared
        
        elif delta_share_V == True and delta_share_U == True:
            self.delta_u1 = experts_delta_u1_shared
            self.delta_v1 = experts_delta_v1_shared
            self.delta_u2 = experts_delta_u2_shared
            self.delta_v2 = experts_delta_v2_shared
            self.delta_u3 = experts_delta_u3_shared
            self.delta_v3 = experts_delta_v3_shared

    def update_Wmean(self, Wmean1, Wmean2, Wmean3):
        self.Wmean1.weight.data = Wmean1.weight.data.clone()
        if Wmean1.bias is not None:
            self.Wmean1.bias.data = Wmean1.bias.data.clone()
        self.Wmean2.weight.data = Wmean2.weight.data.clone()
        if Wmean2.bias is not None:
            self.Wmean2.bias.data = Wmean2.bias.data.clone()
        self.Wmean3.weight.data = Wmean3.weight.data.clone()
        if Wmean3.bias is not None:
            self.Wmean3.bias.data = Wmean3.bias.data.clone()

    def update_delta_v(self, delta_v1, delta_v2, delta_v3):
        self.delta_v1.weight.data = delta_v1.weight.data.clone()
        self.delta_v2.weight.data = delta_v2.weight.data.clone()
        self.delta_v3.weight.data = delta_v3.weight.data.clone()

    def update_delta_u(self, delta_u1, delta_u2, delta_u3):
        self.delta_u1.weight.data = delta_u1.weight.data.clone()
        self.delta_u2.weight.data = delta_u2.weight.data.clone()
        self.delta_u3.weight.data = delta_u3.weight.data.clone()

    def forward(self, hidden_states):
        # if self.sparse_w1 == None:
        up = self.Wmean3(hidden_states) + self.delta_u3(self.delta_v3(hidden_states))
        gate = self.Wmean1(hidden_states) + self.delta_u1(self.delta_v1(hidden_states))
        return self.Wmean2(self.act_fn(gate) * up) + self.delta_u2(self.delta_v2(self.act_fn(gate) * up))
        # else:
        #     up = self.Wmean3_U(self.Wmean3_V(hidden_states)) + self.delta_u3(self.delta_v3(hidden_states)) + self.sparse_w3(hidden_states)
        #     gate = self.Wmean1_U(self.Wmean1_V(hidden_states)) + self.delta_u1(self.delta_v1(hidden_states)) + self.sparse_w1(hidden_states)
        #     return self.Wmean2_U(self.Wmean2_V(self.act_fn(gate) * up)) + self.delta_u2(self.delta_v2(self.act_fn(gate) * up)) + self.sparse_w2(self.act_fn(gate) * up)
        
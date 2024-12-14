import torch
import numpy as np
from tqdm import tqdm
import time
import itertools
import os
import sys
from torch.utils.data import DataLoader
import math
from datasets import load_dataset
from accelerate import Accelerator
import random
from .data_utils import get_test_data

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


@torch.no_grad()
def ppl_eval_sharing(model, tokenizer, experiment_name, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=16, params_only=False):
    seed = 42  # or any other integer
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    model.eval()
    ppls = {}
    total_allocated_list = []
    total_reserved_list = []

    # 获取模型的主设备
    main_device = next(model.parameters()).device
    if not params_only:
        for dataset in datasets:
            # 对于其他数据集，使用原有的加载方式
            data = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
            # data = next(iter(data)).to(main_device)  # 假设 get_test_data 返回一个 DataLoader

            seqlen = model_seq_len
            n_samples = len(data)
            nlls = []

            with tqdm(range(n_samples), desc=f"Evaluating {dataset} - Perplexity") as progress_bar:
                for i in progress_bar:
                    batch = next(iter(data)).to(main_device)

                    allocated, reserved = print_memory_usage()
                    total_allocated_list.append(allocated)
                    total_reserved_list.append(reserved)

                    with torch.no_grad():
                        output = model(batch)
                        logits = output.logits if hasattr(output, "logits") else output[0]

                    # 确保 logits 在正确的设备上
                    logits = logits.to(main_device)
                    shift_logits = logits[:, :-1, :].contiguous().float()
                    shift_labels = batch[:, 1:].contiguous()

                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    neg_log_likelihood = loss.float() * seqlen
                    nlls.append(neg_log_likelihood)

                    curr_ppl = _perplexity(nlls, i + 1, seqlen)
                    progress_bar.set_description(f"Evaluating {dataset} - Perplexity {curr_ppl:.3f}")

            ppl = _perplexity(nlls, n_samples, seqlen)
            ppls[dataset] = ppl.item()

    # 计算参数统计
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # non_trainable_params = total_params - trainable_params
    threshold = 1e-6
    non_zero_params = sum((p.abs() > threshold).sum().item() for p in model.parameters())

    # 检查 SVD 压缩的 Mixtral 特性
    # svd_layers = sum(1 for m in model.modules() if isinstance(m, SVD_MixtralSparseMoeBlock))
    print("\n")
    result_str = f"Experiment: {experiment_name}\n"
    if not params_only:
        avg_allocated = sum(total_allocated_list) / len(total_allocated_list)
        avg_reserved = sum(total_reserved_list) / len(total_reserved_list)
        result_str += f"PPL after evaluation: {ppls}\n"
        result_str += f"Average Allocated Memory: {avg_allocated:.2f} MiB\n"
        result_str += f"Average Reserved Memory: {avg_reserved:.2f} MiB\n"
    
    # result_str += f"Total number of parameters: {total_params / 1e9:.2f}B\n"
    # result_str += f"Number of trainable parameters: {trainable_params / 1e9:.2f}B\n"
    # result_str += f"Number of non-trainable parameters: {non_trainable_params / 1e9:.2f}B\n"
    result_str += f"Number of non-zero parameters: {non_zero_params / 1e9:.2f}B\n"
    if "Mixtral" in experiment_name:
        org_params = 46.70e9
        result_str += f"Compression ratio: {1 - (non_zero_params / org_params):.2f}%\n"
        result_str += f"Save ratio: {non_zero_params / org_params:.2f}%\n"

    elif "Llamix" in experiment_name:
        org_params = 0.40e9
        result_str += f"Compression ratio: {1 - (non_zero_params / org_params):.2f}%\n"
        result_str += f"Save ratio: {non_zero_params / org_params:.2f}%\n"


    print(result_str)
    # return result_str

def print_memory_usage():
    total_gpus = torch.cuda.device_count()
    total_allocated = 0
    total_reserved = 0
    
    for i in range(total_gpus):
        allocated = torch.cuda.memory_allocated(device=i) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device=i) / 1024 / 1024
        total_allocated += allocated
        total_reserved += reserved
        # print(f"GPU {i} - Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")
    
    # print(f"Total - Allocated: {total_allocated:.2f} MiB, Reserved: {total_reserved:.2f} MiB")
    
    return total_allocated, total_reserved

@torch.no_grad()
def ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device="cuda"):
    model.eval()
    ppls = {}
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size = batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            output = model(batch, use_cache=False)
            lm_logits = output.logits
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
    print("PPL after pruning: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

# only call this function when for 65b or more model    
@torch.no_grad()
def ppl_eval_large(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], seq_len=2048, batch_size=32, device="cuda"):
    import  torch.nn as nn
    class LlamaRMSNorm(nn.Module):
        def __init__(self, hidden_size=model.config.hidden_size, eps=model.config.rms_norm_eps):
            """
            LlamaRMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
    norm = LlamaRMSNorm().bfloat16().cuda()
    lm_head = model.lm_head.cuda()
    model.eval()
    ppls = {}
    layers = model.model.layers
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            model.model.embed_tokens = model.model.embed_tokens.cuda()
            model.model.norm = model.model.norm.cuda()
            layers[0] = layers[0].cuda()

            dtype = next(iter(model.parameters())).dtype
            inps = torch.zeros(
                (batch.shape[0], seq_len, model.config.hidden_size), dtype=dtype, device="cuda"
            )
            cache = {'i': 0, 'attention_mask': None, "position_ids": None}
            class Catcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, **kwargs):
                    inps[cache['i']] = inp
                    cache['i'] += 1
                    if cache['attention_mask'] is None:
                        cache['attention_mask'] = kwargs['attention_mask']
                        cache['position_ids'] = kwargs['position_ids']
                    else:
                        cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                        cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
                    raise ValueError
            layers[0] = Catcher(layers[0])
            for j in range(batch.shape[0]):
                try:
                    model(batch[j].unsqueeze(0).cuda())
                except ValueError:
                    pass
            layers[0] = layers[0].module
            layers[0] = layers[0].cpu()
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            model.model.norm = model.model.norm.cpu()
            torch.cuda.empty_cache()
            attention_masks = cache['attention_mask']
            position_ids = cache['position_ids']
            for i in range(len(layers)):
                layer = layers[i].cuda()
                outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
                layers[i] = layer.cpu()
                inps = outs
                torch.cuda.empty_cache()
            hidden_states = norm(outs)
            lm_logits = lm_head(hidden_states)
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous().cuda()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
            else:
                print("warning: nan or inf in lm_logits")
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
    print("PPL after pruning: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

@torch.no_grad()
def eff_eval(model, tokenizer, datasets='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda"):
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(datasets, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            throughput += end_time - start_time
            print("time: {}".format(end_time - start_time))
    print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    print("Throughput: {} tokens/sec".format(token_num / throughput))

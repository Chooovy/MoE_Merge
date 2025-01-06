import torch
import time
import itertools
from tqdm import tqdm
import argparse
import os
from component.data_utils import get_test_data
from component.merge_mixtral_keepWmean_scale_delta import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import load_checkpoint_and_dispatch
from torch.utils.flop_counter import FlopCounterMode
from collections import defaultdict
from accelerate import init_empty_weights

cfg = {}

def get_model():
    cfg['merge_model'] = False
    cfg['mask'] = False
    cfg['deepcopy'] = False
    using_Mixtral = False
    cfg['together'] = False

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
                                                    delta_share_V=share_V, delta_share_U=share_U, merge_method=merge_method, deepcopy=cfg['deepcopy']).to(model.model.layers[i].block_sparse_moe.gate.weight.device)

            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            pbar = tqdm(total=len(checkpoint), desc="Loading checkpoint")
            for k, v in checkpoint.items():
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
    
    return model, tokenizer

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128,
             batch_size=1, device='gpu', max_time=600):
    model.eval()
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)

    is_cuda = device == 'gpu'
    if is_cuda:
        devices = [d for d in range(torch.cuda.device_count())]
        weight_memory = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 ** 3)
    else:
        devices = []
        weight_memory = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 ** 3)
    
    # -------------------------------
    # 第一部分：测量吞吐量（Throughput）
    # -------------------------------
    print("开始测量吞吐量...", flush=True)
    num_batches_to_fetch = 5 if device == 'gpu' else 2
    throughput_time = 0
    token_num = 0
    completed_batches = 0

    # 在评测开始前同步设备，清理缓存
    if is_cuda:
        for d in devices:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(d)

    start_time_total = time.perf_counter()

    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        input_device = next(model.parameters()).device if is_cuda else torch.device('cpu')
        batch = batch_data.to(input_device)
        
        # 开始计时
        start_time = time.perf_counter()

        # 生成输出
        generation_output = model.generate(
            input_ids=batch,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            use_cache=True,
            top_k=50,
            max_length=original_len + generated_len,
            top_p=0.95,
            temperature=1,
        )

        # 同步设备，确保所有计算完成
        if is_cuda:
            for d in devices:
                torch.cuda.synchronize(d)

        # 结束计时
        end_time = time.perf_counter()

        batch_time = end_time - start_time
        throughput_time += batch_time
        token_num += batch.shape[0] * generated_len
        completed_batches += 1

        print(f"批次 {batch_idx + 1}/{num_batches_to_fetch} - 耗时: {batch_time:.4f}s - 生成 tokens 数: {batch.shape[0] * generated_len}", flush=True)

        # 检查是否超过最大时间限制
        if end_time - start_time_total > max_time:
            print(f"已达到最大时间限制 {max_time} 秒，停止评估。", flush=True)
            break

    total_time = time.perf_counter() - start_time_total

    if is_cuda:
        # 在评测结束后同步设备，获取内存占用
        for d in devices:
            torch.cuda.synchronize(d)
        current_memory = sum(torch.cuda.max_memory_allocated(d) for d in devices)
        activation_memory = (current_memory) / (1024 ** 3)
        memory_info = f"总内存占用: {current_memory / (1024 ** 3):.2f} GB\n" \
                      f"权重内存: {weight_memory:.2f} GB\n" \
                      f"激活值内存: {activation_memory - weight_memory:.2f} GB\n"
    else:
        memory_info = "在 CPU 上运行，无法获得内存测量数据。\n"

    avg_throughput = token_num / throughput_time if throughput_time > 0 else 0
    throughput_info = f"吞吐量: {avg_throughput:.2f} tokens/sec\n" \
                      f"完成批次数: {completed_batches}/{num_batches_to_fetch}\n" \
                      f"总评估时间: {total_time:.2f} 秒\n" \
                      f"每批次平均时间: {(throughput_time / completed_batches):.2f} 秒\n" \
                      f"生成长度: {generated_len}\n"

    # -------------------------------
    # 第二部分：测量 FLOPs
    # -------------------------------
    print("开始测量 FLOPs...", flush=True)
    total_flops = 0
    flops_completed_batches = 0
    num_batches_to_fetch_flops = 5  # 为了节省时间，FLOPs 测量可使用更少的批次

    # 重新加载数据
    test_loader_flops = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)

    # 同步设备
    if is_cuda:
        for d in devices:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(d)

    start_time_total_flops = time.perf_counter()

    for batch_idx, batch_data in enumerate(itertools.islice(test_loader_flops, num_batches_to_fetch_flops)):
        input_device = next(model.parameters()).device if is_cuda else torch.device('cpu')
        batch = batch_data.to(input_device)

        # 初始化 FLOPs 计数器
        flop_counter = FlopCounterMode(model, display=False)

        # Reset FLOPs count
        flop_counter.flop_counts = defaultdict(lambda: defaultdict(int))

        # 开始计时
        start_time = time.perf_counter()

        with flop_counter:
            generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len + generated_len,
                top_p=0.95,
                temperature=1,
            )

        # 同步设备
        if is_cuda:
            for d in devices:
                torch.cuda.synchronize(d)

        # 结束计时
        end_time = time.perf_counter()

        batch_flops = flop_counter.get_total_flops()
        total_flops += batch_flops
        flops_completed_batches += 1

        batch_time = end_time - start_time
        print(f"批次 {batch_idx + 1}/{num_batches_to_fetch_flops} - FLOPs: {batch_flops:.2e} - Time: {batch_time:.4f}s", flush=True)
        # print(f"批次 {batch_idx + 1}/{num_batches_to_fetch_flops} - FLOPs: {batch_flops:.2e} - 耗时: {batch_time:.4f}s")

        # 检查是否超过最大时间限制
        if end_time - start_time_total_flops > max_time:
            print(f"已达到最大时间限制 {max_time} 秒，停止 FLOPs 评估。", flush=True)
            break

    total_time_flops = time.perf_counter() - start_time_total_flops

    avg_flops = total_flops / flops_completed_batches if flops_completed_batches > 0 else 0
    flops_info = f"平均每批次 FLOPs: {avg_flops:.2e}\n" \
                 f"完成批次数: {flops_completed_batches}/{num_batches_to_fetch_flops}\n" \
                 f"总评估时间: {total_time_flops:.2f} 秒\n" \
                 f"生成长度: {generated_len}\n"

    # 返回结果
    result = memory_info + "\n---- 吞吐量测量结果 ----\n" + throughput_info + "\n---- FLOPs 测量结果 ----\n" + flops_info

    return result



def main(args):
    model, tokenizer = get_model()
    
    result = eff_eval(model, tokenizer, dataset='wikitext2', 
                      original_len=args.original_len, generated_len=args.generated_len, 
                      batch_size=args.batch_size, device=args.device)
    
    # 确定实验类型和文件名
    # model_name = args.model_path.rstrip('/').split('/')[-1]
    # if args.device == 'gpu':
    #     if args.varying == 'batch_size':
    #         exp_type = "Varying Batch Size on GPU"
    #         file_suffix = f"varying_batch_size_gpu_{model_name}.txt"
    #     else:
    #         exp_type = "Varying Sequence Length on GPU"
    #         file_suffix = f"varying_sequence_length_gpu_{model_name}.txt"
    # else:
    #     if args.varying == 'batch_size':
    #         exp_type = "Varying Batch Size on CPU"
    #         file_suffix = f"varying_batch_size_cpu_{model_name}.txt"
    #     else:
    #         exp_type = "Varying Sequence Length on CPU"
    #         file_suffix = f"varying_sequence_length_cpu_{model_name}.txt"

    # # 输出文件名（不增加新参数）
    # file_name = file_suffix

    # # 创建输出目录
    # os.makedirs(args.output_dir, exist_ok=True)
    
    # # 将结果写入文件
    # output_path = os.path.join(args.output_dir, file_name)
    # with open(output_path, 'a') as f:
    #     f.write(f"\n--- Experiment: {exp_type} ---\n")
    #     f.write(f"Model: {args.model_path}\n")
    #     f.write(f"Mode: {args.mode}\n")
    #     f.write(f"Device: {args.device}\n")
    #     f.write(f"Batch Size: {args.batch_size}\n")
    #     f.write(f"Original Length: {args.original_len}\n")
    #     f.write(f"Generated Length: {args.generated_len}\n")
    #     f.write(result)
    #     f.write("\n")

    # print(f"Results written to {output_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficiency Evaluation for Language Models")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"], help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--original_len", type=int, default=1024, help="Original sequence length")
    parser.add_argument("--generated_len", type=int, default=128, help="Generated sequence length")
    # parser.add_argument("--varying", type=str, default="batch_size", choices=["batch_size", "sequence_length"], help="Parameter to vary in the experiment")
    # parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the results")
    
    args = parser.parse_args()
    main(args)
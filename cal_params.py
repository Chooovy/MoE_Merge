def calculate_mixtral_memory(
    batch_size=1,
    sequence_length=2048,
    hidden_size=4096,
    num_layers=32,
    vocab_size=32000,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=14336,
    num_experts=8,
    dtype_size=2,  # fp16 = 2 bytes
    training=False
):
    """
    计算 Mixtral-8x7B 模型显存占用
    """
    # 1. 模型参数显存
    # Embedding 层
    embedding_param = vocab_size * hidden_size
    
    # 每个 Transformer 层参数
    per_layer_params = (
        # 自注意力参数
        hidden_size * hidden_size +  # q_proj
        hidden_size * (hidden_size // 4) * 2 +  # k_proj, v_proj
        hidden_size * hidden_size +  # o_proj
        # MoE 参数
        hidden_size * num_experts +  # gate
        num_experts * (
            hidden_size * intermediate_size * 2 +  # w1, w3
            intermediate_size * hidden_size  # w2
        ) +
        # LayerNorm 参数
        hidden_size * 2  # input_layernorm, post_attention_layernorm
    )
    
    transformer_params = per_layer_params * num_layers
    
    # 输出层参数
    output_params = hidden_size * vocab_size + hidden_size  # lm_head + final_norm
    
    total_params = embedding_param + transformer_params + output_params
    param_memory = total_params * dtype_size / (1024**3)  # 转换为 GB
    
    # 2. 激活值显存
    # 每个 token 的隐藏状态
    hidden_memory = batch_size * sequence_length * hidden_size * dtype_size
    
    # 注意力 key/value 缓存
    kv_cache = batch_size * sequence_length * (hidden_size // 2) * num_layers * dtype_size
    
    activation_memory = (hidden_memory + kv_cache) / (1024**3)  # 转换为 GB
    
    # 3. 如果是训练模式，需要考虑优化器状态
    optimizer_memory = 0
    if training:
        optimizer_memory = total_params * dtype_size * 2 / (1024**3)  # Adam 优化器需要 2 个状态
    
    total_memory = param_memory + activation_memory + optimizer_memory
    
    return {
        "参数显存 (GB)": param_memory,
        "激活值显存 (GB)": activation_memory,
        "优化器显存 (GB)": optimizer_memory,
        "总显存 (GB)": total_memory
    }


# # 推理模式
# print("推理模式:")
# memory_usage = calculate_mixtral_memory(batch_size=1, sequence_length=2048)
# for k, v in memory_usage.items():
#     print(f"{k}: {v:.2f}")

# # 训练模式
# print("\n训练模式:")
# memory_usage = calculate_mixtral_memory(batch_size=1, sequence_length=2048, training=True)
# for k, v in memory_usage.items():
#     print(f"{k}: {v:.2f}")


def calculate_mixtral_params(
    hidden_size=4096,
    num_layers=32,
    vocab_size=32000,
    num_attention_heads=32,
    num_key_value_heads=8,
    intermediate_size=14336,
    num_experts=8
):
    """
    计算 Mixtral-8x7B 模型参数量(单位:B)
    """
    # Embedding 层参数
    embedding_params = vocab_size * hidden_size
    
    # 每个 Transformer 层参数
    per_layer_params = (
        # 自注意力参数
        hidden_size * hidden_size +  # q_proj
        hidden_size * (hidden_size // 4) * 2 +  # k_proj, v_proj (使用 grouped query attention)
        hidden_size * hidden_size +  # o_proj
        
        # MoE 参数
        hidden_size * num_experts +  # gate
        num_experts * (
            hidden_size * intermediate_size * 2 +  # w1, w3
            intermediate_size * hidden_size  # w2
        ) +
        
        # LayerNorm 参数
        hidden_size * 2  # input_layernorm, post_attention_layernorm
    )
    
    # 所有 Transformer 层的参数
    transformer_params = per_layer_params * num_layers
    
    # 输出层参数
    output_params = hidden_size * vocab_size + hidden_size  # lm_head + final_norm
    
    # 总参数量
    total_params = embedding_params + transformer_params + output_params
    
    # 转换为 Billion
    total_params_b = total_params / (1000 ** 3)
    
    # 详细分布
    details = {
        "Embedding 层 (B)": embedding_params / (1000 ** 3),
        "每个 Transformer 层 (B)": per_layer_params / (1000 ** 3),
        "所有 Transformer 层 (B)": transformer_params / (1000 ** 3),
        "输出层 (B)": output_params / (1000 ** 3),
        "总参数量 (B)": total_params_b
    }
    
    return details, total_params_b

# 计算并打印结果
# params, total_params_b = calculate_mixtral_params(hidden_size=768,
#     num_layers=6,
#     vocab_size=32128,
#     intermediate_size=3072,
#     num_experts=8)
# params, total_params_b = calculate_mixtral_params()
# for k, v in params.items():
#     print(f"{k}: {v:.2f}")

def calculate_layer_params(
    hidden_size=4096,
    intermediate_size=14336,
    num_experts=8,
    delta_ratio=0.5,
    share_V=True
):
    # 1. Attention参数量
    attention_params = (
        hidden_size * hidden_size +      # q_proj: 4096 * 4096
        hidden_size * (hidden_size // 4) * 2 +  # k_proj, v_proj: 4096 * (4096/4) * 2
        hidden_size * hidden_size        # o_proj: 4096 * 4096
    )
    
    # 2. MoE参数量
    delta_low_rank = int(hidden_size * intermediate_size * delta_ratio / (hidden_size + intermediate_size))
    
    if share_V:
        moe_params = (
            hidden_size * num_experts +   # gate
            (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size        # Wmean2
            ) +
            (
                hidden_size * delta_low_rank +        # shared delta_v1
                intermediate_size * delta_low_rank +   # shared delta_v2
                hidden_size * delta_low_rank          # shared delta_v3
            ) +
            num_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                delta_low_rank * hidden_size +        # delta_u2
                delta_low_rank * intermediate_size    # delta_u3
            )
        )
    else:
        moe_params = (
            hidden_size * num_experts +   # gate
            (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size        # Wmean2
            ) +
            num_experts * (
                delta_low_rank * intermediate_size +   # delta_u1
                hidden_size * delta_low_rank +        # delta_v1
                delta_low_rank * hidden_size +        # delta_u2
                intermediate_size * delta_low_rank +   # delta_v2
                delta_low_rank * intermediate_size +   # delta_u3
                hidden_size * delta_low_rank          # delta_v3
            )
        )
    
    return {
        "Attention参数量 (B)": attention_params / (1000 ** 3),
        "MoE参数量 (B)": moe_params / (1000 ** 3),
        "总参数量 (B)": (attention_params + moe_params) / (1000 ** 3)
    }

# 计算并打印结果
# params = calculate_layer_params()
# for k, v in params.items():
#     print(f"{k}: {v:.2f}")


def calculate_compressed_mixtral_params(
    hidden_size=4096,
    num_layers=32,
    vocab_size=32000,
    intermediate_size=14336,
    num_experts=8,
    delta_ratio=0.5,
    share_ratio=1,
    share_V=True,
    share_U=False,
    merge_method="mean",
    use_pp=True,
    pp_ratio=0.2
):
    # Embedding 层参数
    embedding_params = vocab_size * hidden_size

    delta_low_rank = int(hidden_size * intermediate_size * delta_ratio / (hidden_size + intermediate_size))
    
    if share_V == False:
        # 每个 Transformer 层参数
        per_layer_params = (
            # 自注意力参数
            hidden_size * hidden_size +  # q_proj
            hidden_size * (hidden_size // 4) * 2 +  # k_proj, v_proj (使用 grouped query attention)
            hidden_size * hidden_size +  # o_proj
            
            # MoE 参数
            hidden_size * num_experts +  # gate
            (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) +
            num_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                hidden_size * delta_low_rank + # delta_v1
                delta_low_rank * hidden_size + # delta_u2
                intermediate_size * delta_low_rank + # delta_v2
                delta_low_rank * intermediate_size +  # delta_u3
                hidden_size * delta_low_rank # delta_v3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    elif share_V == True and use_pp == False:
        per_layer_params = (
            # 自注意力参数
            hidden_size * hidden_size +  # q_proj
            hidden_size * (hidden_size // 4) * 2 +  # k_proj, v_proj (使用 grouped query attention)
            hidden_size * hidden_size +  # o_proj
            
            # MoE 参数
            hidden_size * num_experts +  # gate
            (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) +
            (
                hidden_size * delta_low_rank + # shared delta_v1
                intermediate_size * delta_low_rank + # shared delta_v2
                hidden_size * delta_low_rank # shared delta_v3
            ) +
            num_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                delta_low_rank * hidden_size + # delta_u2
                delta_low_rank * intermediate_size  # delta_u3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    elif share_V == True and use_pp == True:
        per_layer_params = (
            # 自注意力参数
            hidden_size * hidden_size +  # q_proj
            hidden_size * (hidden_size // 4) * 2 +  # k_proj, v_proj (使用 grouped query attention)
            hidden_size * hidden_size +  # o_proj
            
            # MoE 参数
            hidden_size * num_experts +  # gate
            (
                hidden_size * intermediate_size * 2 +  # Wmean1, Wmean3
                intermediate_size * hidden_size  # Wmean2
            ) * (1 - pp_ratio) +
            (
                hidden_size * delta_low_rank + # shared delta_v1
                intermediate_size * delta_low_rank + # shared delta_v2
                hidden_size * delta_low_rank # shared delta_v3
            ) +
            num_experts * (
                delta_low_rank * intermediate_size +  # delta_u1
                delta_low_rank * hidden_size + # delta_u2
                delta_low_rank * intermediate_size  # delta_u3
            ) +
            
            # LayerNorm 参数
            hidden_size * 2  # input_layernorm, post_attention_layernorm
        )
    
    # 所有 Transformer 层的参数
    transformer_params = per_layer_params * num_layers
    
    # 输出层参数
    output_params = hidden_size * vocab_size + hidden_size  # lm_head + final_norm
    
    # 总参数量
    total_params = embedding_params + transformer_params + output_params
    
    # 转换为 Billion
    total_params_b = total_params / (1000 ** 3)
    
    # 详细分布
    details = {
        "Embedding 层 (B)": embedding_params / (1000 ** 3),
        "每个 Transformer 层 (B)": per_layer_params / (1000 ** 3),
        "所有 Transformer 层 (B)": transformer_params / (1000 ** 3),
        "输出层 (B)": output_params / (1000 ** 3),
        "总参数量 (B)": total_params_b
    }
    
    return details, total_params_b


# params, total_params_b = calculate_compressed_mixtral_params(delta_ratio=0.5)
# for k, v in params.items():
#     print(f"{k}: {v:.2f}")

# print(f"Total params: {total_params_b:.2f}B")

def find_delta_ratio(target_compression_ratio=0.2, tolerance=1e-4):
    """
    通过二分查找找到合适的delta_ratio值
    Args:
        target_compression_ratio: 目标压缩比例，如0.2表示压缩20%
        tolerance: 允许的误差范围
    Returns:
        找到的delta_ratio值
    """
    # 获取原始模型参数量
    _, original_params = calculate_mixtral_params()
    target_params = original_params * (1 - target_compression_ratio)
    
    # 二分查找的范围
    left, right = 0.0, 1.0
    
    while right - left > tolerance:
        mid = (left + right) / 2
        _, compressed_params = calculate_compressed_mixtral_params(delta_ratio=mid)
        
        if compressed_params > target_params:
            # 如果当前参数量太大，需要更小的delta_ratio
            right = mid
        else:
            # 如果当前参数量太小，需要更大的delta_ratio
            left = mid
    
    final_ratio = (left + right) / 2
    _, final_params = calculate_compressed_mixtral_params(delta_ratio=final_ratio)
    
    print(f"Original params: {original_params:.2f}B")
    print(f"Target params: {target_params:.2f}B")
    print(f"Achieved params: {final_params:.2f}B")
    print(f"Compression ratio: {(original_params - final_params) / original_params:.2%}")
    
    return final_ratio

# 使用示例
delta_ratio = find_delta_ratio(target_compression_ratio=0.4)
print(f"Found delta_ratio: {delta_ratio:.4f}")
import torch
import cloudpickle

def inspect_model_weights(path):
    """Loads the model state dict and prints the dtype of each weight."""
    # Load the state dict from the file
    saved_dict = torch.load(path)
    # 从字节串反序列化模型
    # model = cloudpickle.loads(saved_dict['model'])

    print(saved_dict[0].keys())

    # Iterate over each key-value pair in the state dict
    # for key, value in saved_dict.items():
    #     # print(f"Key: {key}, dtype: {value.dtype}")
    #     print(f"Key: {key}")


# Path to the .pt file
model_path = "/workspace/guhao_workspace/MoE_Merge/cache/SVD_scale_Mixtral_28_29_30_31_bfloat16.pt"
# model_path = "/workspace/SVD-MOE-new/guhao_debug/0_1_2_3_4_5_6_7_8_9_10_11_12_13_14_15_16_17_18_19_20_21_22_23_24_25_26_27_28_29_30_31both_workspace_SVD_MOE_new_models_Mixtral_profiling_wikitext2_512_42.pt"
# model_path = "/aifs4su/lilujun/SVD-MoE-merge/SVD-MOE-new/svd_debug/0_1_2_3_4_5both_aifs4su_lilujun_SVD_MoE_merge_SmolLlamix_8x101M_profiling_wikitext2_512_42.pt"
# Inspect the model weights
inspect_model_weights(model_path)
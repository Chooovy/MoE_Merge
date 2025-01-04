import torch
import cloudpickle

def inspect_model_weights(path):
    """Loads the model state dict and prints the dtype of each weight."""
    # Load the state dict from the file
    saved_dict = torch.load(path, map_location='cuda:1')
    # 从字节串反序列化模型
    # model = cloudpickle.loads(saved_dict['model'])

    print(saved_dict[0].keys())

    # Iterate over each key-value pair in the state dict
    # for key, value in saved_dict.items():
    #     # print(f"Key: {key}, dtype: {value.dtype}")
    #     print(f"Key: {key}")

# model_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0_1_2_3_32.pt"

# inspect_model_weights(model_path)

import os

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted {path}")



def merge_model(path1, path2, list_index, save_path, use_cuda=False):
    if use_cuda:
        saved_dict1 = torch.load(path1, map_location="cuda:0")
        saved_dict2 = torch.load(path2, map_location="cuda:1")
    else:
        saved_dict1 = torch.load(path1, map_location="cpu")
        saved_dict2 = torch.load(path2, map_location="cpu")

    for i in list_index:
        saved_dict1[i] = saved_dict2[i]
    
    torch.save(saved_dict1, save_path)
    print(f"Model saved to {save_path}")

num_sample = 512

model_path1 = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0_1_2_3_{num_sample}.pt"
model_path2 = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_4_5_6_7_{num_sample}.pt"
save_path = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0-7_{num_sample}.pt"
merge_model(model_path1, model_path2, [4, 5, 6, 7], save_path, use_cuda=False)
remove_file(model_path1)
remove_file(model_path2)


model_path1 = save_path
model_path2 = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_8_9_10_11_{num_sample}.pt"
save_path = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0-11_{num_sample}.pt"
merge_model(model_path1, model_path2, [8, 9, 10, 11], save_path, use_cuda=False)
remove_file(model_path1)
remove_file(model_path2)

model_path1 = save_path
model_path2 = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_12_13_14_15_{num_sample}.pt"
save_path = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0-15_{num_sample}.pt"
merge_model(model_path1, model_path2, [12, 13, 14, 15], save_path, use_cuda=False)
remove_file(model_path1)
remove_file(model_path2)


model_path1 = save_path
model_path2 = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_16_17_18_19_{num_sample}.pt"
save_path = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0-19_{num_sample}.pt"
merge_model(model_path1, model_path2, [16, 17, 18, 19], save_path, use_cuda=False)
remove_file(model_path1)
remove_file(model_path2)

model_path1 = save_path
model_path2 = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_20_21_22_23_{num_sample}.pt"
save_path = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0-23_{num_sample}.pt"
merge_model(model_path1, model_path2, [20, 21, 22, 23], save_path, use_cuda=False)
remove_file(model_path1)
remove_file(model_path2)

model_path1 = save_path
model_path2 = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_24_25_26_27_{num_sample}.pt"
save_path = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0-27_{num_sample}.pt"
merge_model(model_path1, model_path2, [24, 25, 26, 27], save_path, use_cuda=False)
remove_file(model_path1)
remove_file(model_path2)

model_path1 = save_path
model_path2 = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_28_29_30_31_{num_sample}.pt"
save_path = f"/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0-31_{num_sample}.pt"
merge_model(model_path1, model_path2, [28, 29, 30, 31], save_path, use_cuda=False)  
remove_file(model_path1)
remove_file(model_path2)

def to_float16(path1):
    saved_dict1 = torch.load(path1, map_location="cuda:1")
    for i in [0, 1, 2, 3]:
        for key in saved_dict1[i]:
            if torch.is_tensor(saved_dict1[i][key]):
                saved_dict1[i][key] = saved_dict1[i][key].to(torch.bfloat16)
    
    torch.save(saved_dict1, "/aifs4su/lilujun/SVD-MoE-merge/MoE/cache/SVD_scale_Mixtral_0_1_2_3_bfloat16_32.pt")

# to_float16(model_path)


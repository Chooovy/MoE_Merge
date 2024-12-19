import torch
import cloudpickle

def inspect_model_weights(path1, path2):
    saved_dict1 = torch.load(path1)
    saved_dict2 = torch.load(path2)

    for i in [28, 29, 30, 31]:
        saved_dict1[i] = saved_dict2[i]
    
    # for key in saved_dict1:
    #     if torch.is_tensor(saved_dict1[key]):
    #         saved_dict1[key] = saved_dict1[key].float()
    
    torch.save(saved_dict1, "/workspace/guhao_workspace/MoE_Merge/cache/SVD_scale_Mixtral.pt")


# Path to the .pt file
model_path1 = "/workspace/guhao_workspace/MoE_Merge/cache/SVD_scale_Mixtral_0-27_bfloat16.pt"
model_path2 = "/workspace/guhao_workspace/MoE_Merge/cache/SVD_scale_Mixtral_28_29_30_31_bfloat16.pt"

inspect_model_weights(model_path1, model_path2)


def to_float16(path1):
    saved_dict1 = torch.load(path1)
    for i in [28, 29, 30, 31]:
        for key in saved_dict1[i]:
            if torch.is_tensor(saved_dict1[i][key]):
                saved_dict1[i][key] = saved_dict1[i][key].to(torch.bfloat16)
    
    torch.save(saved_dict1, "/workspace/guhao_workspace/MoE_Merge/cache/SVD_scale_Mixtral_28_29_30_31_bfloat16.pt")

# to_float16(model_path1)
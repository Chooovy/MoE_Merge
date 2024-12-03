import torch

def inspect_model_weights(path):
    """Loads the model state dict and prints the dtype of each weight."""
    # Load the state dict from the file
    state_dict = torch.load(path, map_location='cuda:0')
    
    # Iterate over each key-value pair in the state dict
    for key, value in state_dict.items():
        print(f"Key: {key}, dtype: {value.dtype}")

# Path to the .pt file
model_path = "/aifs4su/lilujun/SVD-MoE-merge/MoE/SmolLlamix-8x101M_0.5_svd_delta_merged.pt"

# Inspect the model weights
inspect_model_weights(model_path)
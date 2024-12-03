from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from evaluater import ppl_eval

# path = "/aifs4su/gov/models/Mixtral-8x7B-v0.1"
# path = "/aifs4su/lilujun/TinyLLama-4x1.1B-MoE"  # 4 experts
path = "/aifs4su/gov/models/MiniCPM-MoE-8x2B"  # 8 experts


model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True, 
                                             torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
# ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=32, device="cuda")


print(model)
for i in range(40):
    average_w1 = []
    average_w2 = []
    average_w3 = []

    for j in range(8):
        average_w1.append(model.model.layers[i].mlp.experts[j].w1.weight)
        average_w2.append(model.model.layers[i].mlp.experts[j].w2.weight)
        average_w3.append(model.model.layers[i].mlp.experts[j].w3.weight)

    mean_w1 = sum(average_w1) / len(average_w1)
    mean_w2 = sum(average_w2) / len(average_w2)
    mean_w3 = sum(average_w3) / len(average_w3)

    differences_w1 = [w1 - mean_w1 for w1 in average_w1]
    differences_w2 = [w2 - mean_w2 for w2 in average_w2]
    differences_w3 = [w3 - mean_w3 for w3 in average_w3]

    del average_w1, average_w2, average_w3

    def svd_delta(W, ratio=0.5):
        U, S, VT = torch.linalg.svd(W.float(), full_matrices=False)
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        truc_s = S[:num_s_after_trunc]
        truc_u = U[:, :num_s_after_trunc]
        truc_v = VT[:num_s_after_trunc, :]
        truc_sigma = torch.diag(truc_s)
        # #### Replace Attn, MLP ####
        # sqrtSigma = torch.sqrt(truc_sigma)
        # svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
        # svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
        result = truc_u @ truc_sigma @ truc_v
        return result.to(torch.float16)

    for j in range(8):
        model.model.layers[i].mlp.experts[j].w1.weight.data = svd_delta(differences_w1[j]) + mean_w1
        model.model.layers[i].mlp.experts[j].w2.weight.data = svd_delta(differences_w2[j]) + mean_w2
        model.model.layers[i].mlp.experts[j].w3.weight.data = svd_delta(differences_w3[j]) + mean_w3

    del differences_w1, differences_w2, differences_w3, mean_w1, mean_w2, mean_w3

ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048, batch_size=2, device="cuda")

# PPL before svd: {'wikitext2': 6.569596978661885}
# Weight Memory: 28195.03955078125 MiB

# PPL after 4 layers svd: {'wikitext2': 6.595341103196019}
# Weight Memory: 28193.66455078125 MiB

# PPL after all layers 0.9 compression rate svd: {'wikitext2': 6.96248367540365}
# Weight Memory: 28193.66455078125 MiB

# PPL after all layers 0.5 compression rate svd: {'wikitext2': 11.086874788323975}
# Weight Memory: 14862.771484375 MiB
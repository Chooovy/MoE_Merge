V = vocab_size = 32000
H = hidden_size = 4096
E = num_experts = 8
M = mlp_intermediate_size  = 14336
A = attention_intermediate_size = 1024
L = num_layers = 32


ratio = 0.25
trunc_ratio = (H*M*ratio)/(H+M)
# print(f"trunc_ratio: {trunc_ratio}")

P_Exp = 3*H*M
P_Exp_svd = 3*(H*M + H*M*ratio*8)
P_Rou = H*E
P_Att = 2*(H*H+H*A)


P_svd = V*H + 32*(P_Exp_svd + P_Rou + P_Att + 2*H) + H + V*H
P_All = V*H + 32*(8*P_Exp + P_Rou + P_Att + 2*H) + H + V*H


print(f"P_All: {P_All}")
print(f"P_svd: {P_svd}")
print(f"ratio: {P_svd/P_All}")


target_ratio = 0.5

# 创建一个函数来计算给定ratio下的实际比例
def calculate_actual_ratio(r):
    P_Exp = 3*H*M
    P_Exp_svd = 3*(H*M + H*M*r*8)
    P_Rou = H*E
    P_Att = 2*(H*H+H*A)
    
    P_svd = V*H + 32*(P_Exp_svd + P_Rou + P_Att + 2*H) + H + V*H
    P_All = V*H + 32*(8*P_Exp + P_Rou + P_Att + 2*H) + H + V*H
    
    return P_svd/P_All - target_ratio

# 使用二分查找来找到合适的ratio
def binary_search(low, high, tolerance=1e-6):
    while high - low > tolerance:
        mid = (low + high) / 2
        if calculate_actual_ratio(mid) > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2

# 计算符合条件的ratio
ratio = binary_search(0, 1)


print(f"Required ratio for P_svd/P_All = 0.5: {ratio}")
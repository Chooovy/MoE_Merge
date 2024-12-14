delta_W_{id} = W_{id} - W_mean

W_{id} * scale_{id} * scale_{id}^-1 - W_mean * scale_{id} * scale_{id}^-1 = delta_W_{id} * scale_{id} * scale_{id}^-1

SVD(W_{id} * scale_{id}) = SVD(W_mean * scale_{id} + delta_W_{id} * scale_{id})

W_mean = (W_{id} * scale_{id} + W_{id+1} * scale_{id+1}) / 2



SVD(X) = U * Sigma * V
原本的操作是
W_mean = (W1 + W2) / 2
W1 = SVD(W_mean) + SVD(W1 - W_mean)
W2 = SVD(W_mean) + SVD(W2 - W_mean)
我现在有W1，W2，scale1，scale2
W1 = SVD(W_mean) + SVD((W1 - W_mean)*scale1)*scale1^-1
W2 = SVD(W_mean) + SVD((W2 - W_mean)*scale2)*scale2^-1
从而弥补误差
我想把scale1,scale2应用到W_mean上该如何操作

我想将scale1,scale2使用SVD(W_mean * scale_new) * scale_new^-1的方式应用到W_mean上


我将W1分解为U1,V1并且truncate，W2分解为U2,V2并且truncate
SVD(W1) = U1[:r]*V1[:r]
SVD(W2) = U2[:r]*V2[:r]
这时我将W1，W2合并起来进行SVD分解得到U_new,V_new
SVD(concat[W1, W2]) = U_new[:r]*V_new[:r]
那么原本的U1，V1如何由U_new,V_new得到
U1 = ? 
V1 = ? 
U2 = ? 
V2 = ?
import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True)
def relu(x):
    return np.maximum(0.0, x)

@njit(fastmath=True, cache=True)
def tanh(x):
    return np.tanh(x)

@njit(fastmath=True, cache=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit(fastmath=True, cache=True)
def linear(x, weight, bias):
    # np.dotはNumbaによってBLAS呼び出しにコンパイルされます
    return np.dot(x, weight.T) + bias

@njit(cache=True)
def flatten(x):
    return x.reshape(x.shape[0], -1)

@njit(parallel=True, fastmath=True, cache=True)
def softmax(x):
    # バッチ(x.shape[0])単位で並列化
    out = np.empty_like(x)
    for i in prange(x.shape[0]):
        x_row = x[i]
        x_max = np.max(x_row)
        exps = np.exp(x_row - x_max)
        out[i] = exps / np.sum(exps)
    return out

@njit(parallel=True, fastmath=True, cache=True)
def conv1d(x, weight, bias, stride):
    n_x, c_in, l_in = x.shape
    c_out, _, k = weight.shape
    l_out = (l_in - k) // stride + 1
    
    out = np.empty((n_x, c_out, l_out), dtype=x.dtype)
    
    # バッチ(n_x)で並列化
    for n in prange(n_x):
        for c in range(c_out):
            for l in range(l_out):
                l_start = l * stride
                
                # 畳み込み演算
                val = 0.0
                for ci in range(c_in):
                    for ki in range(k):
                        val += x[n, ci, l_start + ki] * weight[c, ci, ki]
                
                out[n, c, l] = val + bias[c]
    return out

@njit(parallel=True, fastmath=True, cache=True)
def conv2d(x, weight, bias, stride=(1, 1)):
    n_x, c_in, h_in, w_in = x.shape
    c_out, _, k_h, k_w = weight.shape
    s_h, s_w = stride
    
    h_out = (h_in - k_h) // s_h + 1
    w_out = (w_in - k_w) // s_w + 1
    
    out = np.empty((n_x, c_out, h_out, w_out), dtype=x.dtype)
    
    # バッチ(n_x)で並列化
    for n in prange(n_x):
        for c in range(c_out):
            for h in range(h_out):
                for w in range(w_out):
                    h_start = h * s_h
                    w_start = w * s_w
                    
                    val = 0.0
                    for ci in range(c_in):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                val += x[n, ci, h_start + kh, w_start + kw] * weight[c, ci, kh, kw]
                    
                    out[n, c, h, w] = val + bias[c]
    return out

@njit(parallel=True, fastmath=True, cache=True)
def max_pool2d(x, kernel_size=(2, 2), stride=(2, 2)):
    n_x, c, h_in, w_in = x.shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    
    h_out = (h_in - k_h) // s_h + 1
    w_out = (w_in - k_w) // s_w + 1
    
    out = np.empty((n_x, c, h_out, w_out), dtype=x.dtype)
    
    # バッチ(n_x)とチャネル(c)の両方で並列化
    for n in prange(n_x):
        for ci in range(c):
            for h in range(h_out):
                for w in range(w_out):
                    h_start = h * s_h
                    w_start = w * s_w
                    
                    # ウィンドウ内の最大値を探す
                    max_val = x[n, ci, h_start, w_start]
                    for kh in range(k_h):
                        for kw in range(k_w):
                            val = x[n, ci, h_start + kh, w_start + kw]
                            if val > max_val:
                                max_val = val
                    
                    out[n, ci, h, w] = max_val
    return out
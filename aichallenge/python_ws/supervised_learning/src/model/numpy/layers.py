import numpy as np
from numpy.lib.stride_tricks import as_strided

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def linear(x, weight, bias):
    return np.dot(x, weight.T) + bias

def conv1d(x, weight, bias, stride):
    n_x, c_in, l_in = x.shape
    c_out, _, k = weight.shape
    l_out = (l_in - k) // stride + 1
    s0, s1, s2 = x.strides
    strided_x = as_strided(x, shape=(n_x, c_in, l_out, k), strides=(s0, s1, s2 * stride, s2))
    strided_x_reshaped = strided_x.transpose(0, 2, 1, 3).reshape(n_x * l_out, c_in * k)
    weight_reshaped = weight.reshape(c_out, -1)
    conv_val = strided_x_reshaped @ weight_reshaped.T
    conv_val_reshaped = conv_val.reshape(n_x, l_out, c_out).transpose(0, 2, 1)
    return conv_val_reshaped + bias.reshape(1, -1, 1)

def conv2d(x, weight, bias, stride=(1, 1)):
    n_x, c_in, h_in, w_in = x.shape
    c_out, _, k_h, k_w = weight.shape
    s_h, s_w = stride

    h_out = (h_in - k_h) // s_h + 1
    w_out = (w_in - k_w) // s_w + 1
    
    s0, s1, s2, s3 = x.strides
    
    strided_x = as_strided(x,
                           shape=(n_x, c_in, h_out, w_out, k_h, k_w),
                           strides=(s0, s1, s2 * s_h, s3 * s_w, s2, s3))
    
    strided_x_reshaped = strided_x.transpose(0, 2, 3, 1, 4, 5).reshape(n_x * h_out * w_out, c_in * k_h * k_w)
    
    weight_reshaped = weight.reshape(c_out, -1)
    
    conv_val = strided_x_reshaped @ weight_reshaped.T
    
    conv_val_reshaped = conv_val.reshape(n_x, h_out, w_out, c_out)
    
    conv_val_final = conv_val_reshaped.transpose(0, 3, 1, 2)
    
    return conv_val_final + bias.reshape(1, -1, 1, 1)

def max_pool2d(x, kernel_size=(2, 2), stride=(2, 2)):
    n_x, c, h_in, w_in = x.shape
    k_h, k_w = kernel_size
    s_h, s_w = stride
    
    h_out = (h_in - k_h) // s_h + 1
    w_out = (w_in - k_w) // s_w + 1
    
    s0, s1, s2, s3 = x.strides
    
    strided_x = as_strided(x,
                           shape=(n_x, c, h_out, w_out, k_h, k_w),
                           strides=(s0, s1, s2 * s_h, s3 * s_w, s2, s3))
    
    return strided_x.max(axis=(4, 5))

def flatten(x):
    return x.reshape(x.shape[0], -1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - x_max)
    return exps / np.sum(exps, axis=1, keepdims=True)
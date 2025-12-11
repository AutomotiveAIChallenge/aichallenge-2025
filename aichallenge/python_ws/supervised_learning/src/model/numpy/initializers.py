import numpy as np

def kaiming_normal_init(shape, fan_out):
    """Kaiming (He) Normal初期化 (mode='fan_out')"""
    std = np.sqrt(2.0 / fan_out)
    return np.random.normal(0, std, size=shape)

def zeros_init(shape):
    """ゼロ初期化"""
    return np.zeros(shape)
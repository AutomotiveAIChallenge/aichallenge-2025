from src.model.numpy.layers import (
    relu,       # NumPy版 (高速だった)
    tanh,       # NumPy版 (高速だった)
    sigmoid,  
    linear,     # NumPy版 (BLASが使える)
    flatten,    # NumPy版 (差がない)
    softmax,    # NumPy版 (高速だった)
    conv1d,     # NumPy版 (高速だった)
    conv2d,     # NumPy版 (高速だった)
    max_pool2d, 
)

from src.model.numba.layers import (
    sigmoid,    # Numba版 (高速だった )
    max_pool2d  # Numba版 (高速だった )
)

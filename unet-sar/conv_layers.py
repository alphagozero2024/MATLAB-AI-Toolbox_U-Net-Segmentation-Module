"""
Convolutional layers for U-Net implementation
Using autograd from np_mlp_autograd
Supports both NumPy (CPU) and CuPy (CUDA) backends
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'np_mlp_autograd'))

import numpy as np
import math
from typing import Tuple, Optional
from np_mlp_autograd.autograd import Tensor, tensor
from np_mlp_autograd.nn import Module, Parameter
from backend import get_backend


class Conv2D(Module):
    """
    2D Convolution layer with autograd support
    Implements: Z^(l+1) = ReLU(W^(l) * Z^(l) + b^(l))
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Kaiming initialization for Conv layers (He initialization)
        fan_in = in_channels * kernel_size * kernel_size
        limit = math.sqrt(2.0 / fan_in)
        
        # Weight shape: (out_channels, in_channels, kernel_size, kernel_size)
        W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit
        self.weight = Parameter(W)
        
        if bias:
            self.bias = Parameter(np.zeros((out_channels,)))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for 2D convolution
        Input: (batch_size, in_channels, height, width)
        Output: (batch_size, out_channels, out_height, out_width)
        """
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)


def conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None, 
           stride: int = 1, padding: int = 0) -> Tensor:
    """
    2D convolution operation with autograd support
    Supports both NumPy (CPU) and CuPy (CUDA) backends
    
    Args:
        x: Input tensor (batch_size, in_channels, height, width)
        weight: Convolution kernel (out_channels, in_channels, kernel_size, kernel_size)
        bias: Bias term (out_channels,)
        stride: Stride for convolution
        padding: Padding for input
    
    Returns:
        Output tensor (batch_size, out_channels, out_height, out_width)
    """
    backend = get_backend()
    xp = backend.xp
    
    x_data = x.data
    w_data = weight.data
    
    # Add padding if needed
    if padding > 0:
        x_data = xp.pad(x_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                       mode='constant', constant_values=0)
    
    batch_size, in_channels, in_h, in_w = x_data.shape
    out_channels, _, k_h, k_w = w_data.shape
    
    # Calculate output dimensions
    out_h = (in_h - k_h) // stride + 1
    out_w = (in_w - k_w) // stride + 1
    
    # Initialize output
    out_data = xp.zeros((batch_size, out_channels, out_h, out_w), dtype=xp.float64)
    
    # Perform convolution
    for b in range(batch_size):
        for oc in range(out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + k_h
                    w_end = w_start + k_w
                    
                    # Extract receptive field
                    receptive_field = x_data[b, :, h_start:h_end, w_start:w_end]
                    # Convolve with kernel
                    out_data[b, oc, i, j] = xp.sum(receptive_field * w_data[oc])
    
    # Create output tensor
    out = Tensor(out_data, requires_grad=x.requires_grad or weight.requires_grad,
                _children=(x, weight), op='conv2d')
    
    # Store variables for backward pass
    x_padded = xp.pad(x.data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                     mode='constant', constant_values=0) if padding > 0 else x.data
    
    def _backward():
        if out.grad is None:
            return
        
        # Gradient w.r.t input
        if x.requires_grad:
            dx_padded = xp.zeros_like(x_padded)
            
            for b in range(batch_size):
                for oc in range(out_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            h_start = i * stride
                            w_start = j * stride
                            h_end = h_start + k_h
                            w_end = w_start + k_w
                            
                            dx_padded[b, :, h_start:h_end, w_start:w_end] += \
                                w_data[oc] * out.grad[b, oc, i, j]
            
            # Remove padding
            if padding > 0:
                dx = dx_padded[:, :, padding:-padding, padding:-padding]
            else:
                dx = dx_padded
            
            x.grad = dx if x.grad is None else x.grad + dx
        
        # Gradient w.r.t weight
        if weight.requires_grad:
            dw = xp.zeros_like(w_data)
            
            for b in range(batch_size):
                for oc in range(out_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            h_start = i * stride
                            w_start = j * stride
                            h_end = h_start + k_h
                            w_end = w_start + k_w
                            
                            receptive_field = x_padded[b, :, h_start:h_end, w_start:w_end]
                            dw[oc] += receptive_field * out.grad[b, oc, i, j]
            
            weight.grad = dw if weight.grad is None else weight.grad + dw
    
    out._backward = _backward
    
    # Add bias if present
    if bias is not None:
        # Reshape bias for broadcasting: (out_channels,) -> (1, out_channels, 1, 1)
        bias_reshaped = bias.data.reshape(1, -1, 1, 1)
        out_with_bias = Tensor(out.data + bias_reshaped, 
                              requires_grad=out.requires_grad or bias.requires_grad,
                              _children=(out, bias), op='conv2d+bias')
        
        def _backward_bias():
            if out_with_bias.grad is None:
                return
            
            # Gradient flows through addition
            if out.requires_grad:
                out.grad = out_with_bias.grad if out.grad is None else out.grad + out_with_bias.grad
                out._backward()
            
            # Gradient w.r.t bias: sum over batch, height, width
            if bias.requires_grad:
                db = np.sum(out_with_bias.grad, axis=(0, 2, 3))
                bias.grad = db if bias.grad is None else bias.grad + db
        
        out_with_bias._backward = _backward_bias
        return out_with_bias
    
    return out


class MaxPool2D(Module):
    """
    2D Max Pooling layer
    Implements: Z_pool^(l) = MaxPool_{2×2}(Z^(l))
    """
    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for max pooling
        Input: (batch_size, channels, height, width)
        Output: (batch_size, channels, out_height, out_width)
        """
        return max_pool2d(x, self.kernel_size, self.stride)


def max_pool2d(x: Tensor, kernel_size: int = 2, stride: Optional[int] = None) -> Tensor:
    """
    2D max pooling operation with autograd support
    """
    backend = get_backend()
    xp = backend.xp
    
    if stride is None:
        stride = kernel_size
    
    x_data = x.data
    batch_size, channels, in_h, in_w = x_data.shape
    
    # Calculate output dimensions
    out_h = (in_h - kernel_size) // stride + 1
    out_w = (in_w - kernel_size) // stride + 1
    
    # Initialize output
    out_data = xp.zeros((batch_size, channels, out_h, out_w), dtype=xp.float64)
    
    # Store max indices for backward pass
    max_indices = xp.zeros((batch_size, channels, out_h, out_w, 2), dtype=xp.int32)
    
    # Perform max pooling
    for b in range(batch_size):
        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + kernel_size
                    w_end = w_start + kernel_size
                    
                    # Extract pooling region
                    pool_region = x_data[b, c, h_start:h_end, w_start:w_end]
                    
                    # Find max value and its position
                    max_val = xp.max(pool_region)
                    max_pos = xp.unravel_index(xp.argmax(pool_region), pool_region.shape)
                    
                    out_data[b, c, i, j] = max_val
                    max_indices[b, c, i, j] = [h_start + max_pos[0], w_start + max_pos[1]]
    
    out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), op='maxpool2d')
    
    def _backward():
        if out.grad is None or not x.requires_grad:
            return
        
        dx = xp.zeros_like(x_data)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        max_h, max_w = max_indices[b, c, i, j]
                        dx[b, c, max_h, max_w] += out.grad[b, c, i, j]
        
        x.grad = dx if x.grad is None else x.grad + dx
    
    out._backward = _backward
    return out


class TransposeConv2D(Module):
    """
    2D Transposed Convolution (Up-convolution) layer
    Implements: Z_up^(l) = W_transpose^(l) ⋆ Z^(l) + b_transpose^(l)
    Used for upsampling in the decoder path
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, 
                 stride: int = 2, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Kaiming initialization
        fan_in = in_channels * kernel_size * kernel_size
        limit = math.sqrt(2.0 / fan_in)
        
        # Weight shape: (in_channels, out_channels, kernel_size, kernel_size)
        W = np.random.randn(in_channels, out_channels, kernel_size, kernel_size) * limit
        self.weight = Parameter(W)
        
        if bias:
            self.bias = Parameter(np.zeros((out_channels,)))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for transposed convolution
        Input: (batch_size, in_channels, height, width)
        Output: (batch_size, out_channels, out_height, out_width)
        """
        return transpose_conv2d(x, self.weight, self.bias, self.stride, self.padding)


def transpose_conv2d(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
                     stride: int = 2, padding: int = 0) -> Tensor:
    """
    2D transposed convolution (deconvolution) with autograd support
    """
    backend = get_backend()
    xp = backend.xp
    
    x_data = x.data
    w_data = weight.data
    
    batch_size, in_channels, in_h, in_w = x_data.shape
    _, out_channels, k_h, k_w = w_data.shape
    
    # Calculate output dimensions
    out_h = (in_h - 1) * stride + k_h - 2 * padding
    out_w = (in_w - 1) * stride + k_w - 2 * padding
    
    # Initialize output
    out_data = xp.zeros((batch_size, out_channels, out_h, out_w), dtype=xp.float64)
    
    # Perform transposed convolution
    for b in range(batch_size):
        for ic in range(in_channels):
            for i in range(in_h):
                for j in range(in_w):
                    h_start = i * stride
                    w_start = j * stride
                    
                    # Add kernel contribution at each position
                    for oc in range(out_channels):
                        out_data[b, oc, h_start:h_start+k_h, w_start:w_start+k_w] += \
                            x_data[b, ic, i, j] * w_data[ic, oc]
    
    # Apply padding (crop)
    if padding > 0:
        out_data = out_data[:, :, padding:-padding, padding:-padding]
    
    out = Tensor(out_data, requires_grad=x.requires_grad or weight.requires_grad,
                _children=(x, weight), op='transpose_conv2d')
    
    def _backward():
        if out.grad is None:
            return
        
        # Add padding to gradient if needed
        grad_padded = out.grad
        if padding > 0:
            grad_padded = xp.pad(out.grad, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                               mode='constant', constant_values=0)
        
        # Gradient w.r.t input
        if x.requires_grad:
            dx = xp.zeros_like(x_data)
            
            for b in range(batch_size):
                for ic in range(in_channels):
                    for i in range(in_h):
                        for j in range(in_w):
                            h_start = i * stride
                            w_start = j * stride
                            
                            for oc in range(out_channels):
                                dx[b, ic, i, j] += xp.sum(
                                    grad_padded[b, oc, h_start:h_start+k_h, w_start:w_start+k_w] * 
                                    w_data[ic, oc]
                                )
            
            x.grad = dx if x.grad is None else x.grad + dx
        
        # Gradient w.r.t weight
        if weight.requires_grad:
            dw = xp.zeros_like(w_data)
            
            for b in range(batch_size):
                for ic in range(in_channels):
                    for i in range(in_h):
                        for j in range(in_w):
                            h_start = i * stride
                            w_start = j * stride
                            
                            for oc in range(out_channels):
                                dw[ic, oc] += x_data[b, ic, i, j] * \
                                    grad_padded[b, oc, h_start:h_start+k_h, w_start:w_start+k_w]
            
            weight.grad = dw if weight.grad is None else weight.grad + dw
    
    out._backward = _backward
    
    # Add bias if present
    if bias is not None:
        bias_reshaped = bias.data.reshape(1, -1, 1, 1)
        out_with_bias = Tensor(out.data + bias_reshaped,
                              requires_grad=out.requires_grad or bias.requires_grad,
                              _children=(out, bias), op='transpose_conv2d+bias')
        
        def _backward_bias():
            if out_with_bias.grad is None:
                return
            
            if out.requires_grad:
                out.grad = out_with_bias.grad if out.grad is None else out.grad + out_with_bias.grad
                out._backward()
            
            if bias.requires_grad:
                db = xp.sum(out_with_bias.grad, axis=(0, 2, 3))
                bias.grad = db if bias.grad is None else bias.grad + db
        
        out_with_bias._backward = _backward_bias
        return out_with_bias
    
    return out


class BatchNorm2D(Module):
    """
    Batch Normalization for 2D data
    Optional component for improved training stability
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        backend = get_backend()
        xp = backend.xp
        
        # Learnable parameters
        self.gamma = Parameter(xp.ones((num_features,)))
        self.beta = Parameter(xp.zeros((num_features,)))
        
        # Running statistics (not learnable)
        self.running_mean = xp.zeros((num_features,))
        self.running_var = xp.ones((num_features,))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for batch normalization
        Input: (batch_size, num_features, height, width)
        """
        backend = get_backend()
        xp = backend.xp
        
        if self.training:
            # Calculate batch statistics
            mean = xp.mean(x.data, axis=(0, 2, 3), keepdims=False)
            var = xp.var(x.data, axis=(0, 2, 3), keepdims=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        mean_bc = mean.reshape(1, -1, 1, 1)
        var_bc = var.reshape(1, -1, 1, 1)
        x_norm = (x.data - mean_bc) / xp.sqrt(var_bc + self.eps)
        
        # Scale and shift
        gamma_bc = self.gamma.data.reshape(1, -1, 1, 1)
        beta_bc = self.beta.data.reshape(1, -1, 1, 1)
        out_data = gamma_bc * x_norm + beta_bc
        
        out = Tensor(out_data, requires_grad=x.requires_grad or self.gamma.requires_grad or self.beta.requires_grad,
                    _children=(x, self.gamma, self.beta), op='batchnorm2d')
        
        # Store for backward
        std = xp.sqrt(var + self.eps)
        
        def _backward():
            if out.grad is None:
                return
            
            N = x.data.shape[0] * x.data.shape[2] * x.data.shape[3]
            
            # Gradient w.r.t gamma
            if self.gamma.requires_grad:
                dgamma = xp.sum(out.grad * x_norm, axis=(0, 2, 3))
                self.gamma.grad = dgamma if self.gamma.grad is None else self.gamma.grad + dgamma
            
            # Gradient w.r.t beta
            if self.beta.requires_grad:
                dbeta = xp.sum(out.grad, axis=(0, 2, 3))
                self.beta.grad = dbeta if self.beta.grad is None else self.beta.grad + dbeta
            
            # Gradient w.r.t input (simplified)
            if x.requires_grad:
                gamma_bc = self.gamma.data.reshape(1, -1, 1, 1)
                std_bc = std.reshape(1, -1, 1, 1)
                dx = gamma_bc * out.grad / std_bc
                x.grad = dx if x.grad is None else x.grad + dx
        
        out._backward = _backward
        return out

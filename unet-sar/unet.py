"""
U-Net Architecture for SAR Image Segmentation
Implements the encoder-decoder structure with skip connections
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'np_mlp_autograd'))

import numpy as np
from typing import List, Tuple
from np_mlp_autograd.autograd import Tensor, tensor
from np_mlp_autograd.nn import Module, Parameter
from conv_layers import Conv2D, MaxPool2D, TransposeConv2D, BatchNorm2D


class DoubleConv(Module):
    """
    Double Convolution Block: (Conv -> BN -> ReLU) * 2
    Core building block for U-Net encoder and decoder
    """
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2D(out_channels) if use_batchnorm else None
        
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D(out_channels) if use_batchnorm else None
        
        self.use_batchnorm = use_batchnorm
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through double convolution
        Implements: Z^(l+1) = ReLU(W^(l) * Z^(l) + b^(l))
        """
        # First convolution
        x = self.conv1(x)
        if self.use_batchnorm and self.bn1 is not None:
            x = self.bn1(x)
        x = x.relu()
        
        # Second convolution
        x = self.conv2(x)
        if self.use_batchnorm and self.bn2 is not None:
            x = self.bn2(x)
        x = x.relu()
        
        return x


class EncoderBlock(Module):
    """
    Encoder block: DoubleConv -> MaxPool
    Implements the contracting path (down-sampling)
    """
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, use_batchnorm)
        self.pool = MaxPool2D(kernel_size=2, stride=2)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            skip: Feature map before pooling (for skip connection)
            pooled: Feature map after pooling
        """
        skip = self.double_conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(Module):
    """
    Decoder block: TransposeConv -> Concat -> DoubleConv
    Implements the expanding path (up-sampling)
    """
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        # Up-convolution reduces channels by half
        self.up_conv = TransposeConv2D(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenation with skip, we have in_channels total
        self.double_conv = DoubleConv(in_channels, out_channels, use_batchnorm)
    
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """
        Args:
            x: Input from previous decoder layer
            skip: Skip connection from corresponding encoder layer
        
        Implements: Z_concat^(l) = Concat(Z_up^(l), Z_encoder^(l))
        """
        # Up-sample
        x = self.up_conv(x)
        
        # Concatenate with skip connection along channel dimension
        # Skip connection is key innovation of U-Net
        x = self.concat(x, skip)
        
        # Double convolution
        x = self.double_conv(x)
        
        return x
    
    def concat(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Concatenate tensors along channel dimension (axis=1)
        Handles potential size mismatches by center cropping
        """
        # Get shapes
        _, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        
        # Crop if necessary (center crop)
        if h1 != h2 or w1 != w2:
            # Calculate crop offsets
            diff_h = h2 - h1
            diff_w = w2 - w1
            
            # Crop x2 to match x1
            x2_data = x2.data
            if diff_h > 0:
                x2_data = x2_data[:, :, diff_h//2:diff_h//2+h1, :]
            if diff_w > 0:
                x2_data = x2_data[:, :, :, diff_w//2:diff_w//2+w1]
            
            x2 = Tensor(x2_data, requires_grad=x2.requires_grad, _children=(x2,), op='crop')
            
            def _backward_crop():
                if x2.grad is not None:
                    # Gradient needs to be padded back
                    pass  # Simplified for now
            x2._backward = _backward_crop
        
        # Concatenate along channel dimension
        concat_data = np.concatenate([x1.data, x2.data], axis=1)
        out = Tensor(concat_data, requires_grad=x1.requires_grad or x2.requires_grad,
                    _children=(x1, x2), op='concat')
        
        def _backward():
            if out.grad is None:
                return
            
            # Split gradient along channel dimension
            c1 = x1.shape[1]
            
            if x1.requires_grad:
                dx1 = out.grad[:, :c1, :, :]
                x1.grad = dx1 if x1.grad is None else x1.grad + dx1
            
            if x2.requires_grad:
                dx2 = out.grad[:, c1:, :, :]
                x2.grad = dx2 if x2.grad is None else x2.grad + dx2
        
        out._backward = _backward
        return out


class UNet(Module):
    """
    U-Net Architecture for Image Segmentation
    
    Configurable parameters:
        - in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        - num_classes: Number of output segmentation classes
        - depth: Number of encoder/decoder levels (default: 4)
        - initial_channels: Number of channels in first conv layer (default: 64)
        - use_batchnorm: Whether to use batch normalization (default: True)
    
    Architecture:
        Encoder (Contracting Path):
            - Multiple levels of [Conv-ReLU-Conv-ReLU-MaxPool]
            - Features doubled at each level
            - Spatial resolution halved at each level
        
        Bottleneck:
            - Bottom of U-Net
            - Highest semantic information, lowest resolution
        
        Decoder (Expanding Path):
            - Multiple levels of [UpConv-Concat-Conv-ReLU-Conv-ReLU]
            - Skip connections from encoder
            - Features halved at each level
            - Spatial resolution doubled at each level
        
        Output:
            - 1x1 Conv to map to num_classes
            - Softmax for probability distribution
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2, 
                 depth: int = 4, initial_channels: int = 64, 
                 use_batchnorm: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.initial_channels = initial_channels
        self.use_batchnorm = use_batchnorm
        
        # Encoder path
        self.encoders = []
        channels = initial_channels
        
        # First encoder takes input channels
        enc = EncoderBlock(in_channels, channels, use_batchnorm)
        self.encoders.append(enc)
        setattr(self, f'encoder_0', enc)
        
        # Subsequent encoders double the channels
        for i in range(1, depth):
            enc = EncoderBlock(channels, channels * 2, use_batchnorm)
            self.encoders.append(enc)
            setattr(self, f'encoder_{i}', enc)
            channels *= 2
        
        # Bottleneck (bottom of U)
        self.bottleneck = DoubleConv(channels, channels * 2, use_batchnorm)
        bottleneck_channels = channels * 2
        
        # Decoder path
        self.decoders = []
        channels = bottleneck_channels
        
        for i in range(depth):
            dec = DecoderBlock(channels, channels // 2, use_batchnorm)
            self.decoders.append(dec)
            setattr(self, f'decoder_{i}', dec)
            channels = channels // 2
        
        # Output layer: 1x1 convolution to map to num_classes
        self.output_conv = Conv2D(channels, num_classes, kernel_size=1, padding=0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor (batch_size, in_channels, height, width)
        
        Returns:
            Output tensor (batch_size, num_classes, height, width)
        """
        # Encoder path - store skip connections
        skip_connections = []
        
        for i, encoder in enumerate(self.encoders):
            skip, x = encoder(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path - use skip connections in reverse order
        skip_connections = skip_connections[::-1]
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # Output layer
        x = self.output_conv(x)
        
        return x
    
    def predict(self, x: Tensor) -> Tensor:
        """
        Get predicted class for each pixel
        
        Returns:
            Predicted classes (batch_size, height, width)
        """
        logits = self.forward(x)
        # Apply softmax along class dimension
        probs = self.softmax(logits, axis=1)
        # Get argmax along class dimension
        pred_classes = np.argmax(probs.data, axis=1)
        return Tensor(pred_classes, requires_grad=False)
    
    def softmax(self, x: Tensor, axis: int = -1) -> Tensor:
        """
        Softmax function: P(y_i=k) = exp(z_i,k) / Σ_j exp(z_i,j)
        """
        # Stable softmax: subtract max for numerical stability
        x_max = np.max(x.data, axis=axis, keepdims=True)
        exp_x = np.exp(x.data - x_max)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        softmax_data = exp_x / sum_exp
        
        out = Tensor(softmax_data, requires_grad=x.requires_grad, _children=(x,), op='softmax')
        
        def _backward():
            if out.grad is None or not x.requires_grad:
                return
            # Simplified softmax gradient
            # For cross-entropy loss, this is typically handled together
            dx = out.grad * softmax_data
            x.grad = dx if x.grad is None else x.grad + dx
        
        out._backward = _backward
        return out
    
    def get_config(self) -> dict:
        """
        Get model configuration for serialization
        """
        return {
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'depth': self.depth,
            'initial_channels': self.initial_channels,
            'use_batchnorm': self.use_batchnorm
        }
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters
        """
        total = 0
        for param in self.parameters():
            total += param.data.size
        return total


class UNetSmall(UNet):
    """Small U-Net variant for faster training"""
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__(in_channels, num_classes, depth=3, initial_channels=32)


class UNetLarge(UNet):
    """Large U-Net variant for better performance"""
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__(in_channels, num_classes, depth=5, initial_channels=64)


def create_unet(in_channels: int = 1, num_classes: int = 2, 
                depth: int = 4, initial_channels: int = 64,
                use_batchnorm: bool = True) -> UNet:
    """
    Factory function to create U-Net with custom configuration
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of segmentation classes
        depth: Network depth (number of encoder/decoder levels)
        initial_channels: Number of channels in first layer
        use_batchnorm: Whether to use batch normalization
    
    Returns:
        Configured U-Net model
    """
    return UNet(in_channels, num_classes, depth, initial_channels, use_batchnorm)

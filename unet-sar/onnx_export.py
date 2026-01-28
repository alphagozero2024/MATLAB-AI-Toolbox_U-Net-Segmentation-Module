"""
ONNX Export Module for U-Net
Converts trained U-Net models to ONNX format for MATLAB integration
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'np_mlp_autograd'))

import numpy as np
from typing import Dict, List, Any, Tuple
import json


try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX library not available. Install with: pip install onnx")


class ONNXExporter:
    """
    Export U-Net model to ONNX format
    
    ONNX (Open Neural Network Exchange) is an open format for representing
    deep learning models, enabling interoperability between frameworks.
    This allows the trained model to be used in MATLAB.
    """
    
    def __init__(self, model, input_shape: Tuple[int, int, int, int]):
        """
        Args:
            model: Trained U-Net model
            input_shape: Input tensor shape (batch_size, channels, height, width)
        """
        self.model = model
        self.input_shape = input_shape
        self.nodes = []
        self.initializers = []
        self.value_info = []
        self.node_counter = 0
    
    def _get_unique_name(self, prefix: str) -> str:
        """Generate unique name for ONNX nodes"""
        name = f"{prefix}_{self.node_counter}"
        self.node_counter += 1
        return name
    
    def _add_conv2d(self, input_name: str, conv_layer, output_name: str):
        """Add Conv2D layer to ONNX graph"""
        if not ONNX_AVAILABLE:
            return
        
        # Get weight and bias
        weight = conv_layer.weight.data
        weight_name = f"{output_name}_weight"
        
        # Create weight initializer
        weight_tensor = numpy_helper.from_array(weight.astype(np.float32), weight_name)
        self.initializers.append(weight_tensor)
        
        # Prepare attributes
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride
        padding = conv_layer.padding
        
        inputs = [input_name, weight_name]
        
        # Add bias if present
        if conv_layer.bias is not None:
            bias = conv_layer.bias.data
            bias_name = f"{output_name}_bias"
            bias_tensor = numpy_helper.from_array(bias.astype(np.float32), bias_name)
            self.initializers.append(bias_tensor)
            inputs.append(bias_name)
        
        # Create Conv node
        conv_node = helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[output_name],
            kernel_shape=[kernel_size, kernel_size],
            strides=[stride, stride],
            pads=[padding, padding, padding, padding],
            name=output_name
        )
        self.nodes.append(conv_node)
    
    def _add_relu(self, input_name: str, output_name: str):
        """Add ReLU activation to ONNX graph"""
        if not ONNX_AVAILABLE:
            return
        
        relu_node = helper.make_node(
            'Relu',
            inputs=[input_name],
            outputs=[output_name],
            name=output_name
        )
        self.nodes.append(relu_node)
    
    def _add_maxpool(self, input_name: str, pool_layer, output_name: str):
        """Add MaxPool2D layer to ONNX graph"""
        if not ONNX_AVAILABLE:
            return
        
        kernel_size = pool_layer.kernel_size
        stride = pool_layer.stride
        
        pool_node = helper.make_node(
            'MaxPool',
            inputs=[input_name],
            outputs=[output_name],
            kernel_shape=[kernel_size, kernel_size],
            strides=[stride, stride],
            name=output_name
        )
        self.nodes.append(pool_node)
    
    def _add_upsample(self, input_name: str, scale: int, output_name: str):
        """Add upsampling operation to ONNX graph"""
        if not ONNX_AVAILABLE:
            return
        
        # Use Resize (recommended) or Upsample
        scales_name = f"{output_name}_scales"
        scales = np.array([1.0, 1.0, float(scale), float(scale)], dtype=np.float32)
        scales_tensor = numpy_helper.from_array(scales, scales_name)
        self.initializers.append(scales_tensor)
        
        # ROI (required for Resize)
        roi_name = f"{output_name}_roi"
        roi = np.array([], dtype=np.float32)
        roi_tensor = numpy_helper.from_array(roi, roi_name)
        self.initializers.append(roi_tensor)
        
        resize_node = helper.make_node(
            'Resize',
            inputs=[input_name, roi_name, scales_name],
            outputs=[output_name],
            mode='nearest',
            name=output_name
        )
        self.nodes.append(resize_node)
    
    def _add_transpose_conv(self, input_name: str, conv_layer, output_name: str):
        """Add TransposeConv2D layer to ONNX graph"""
        if not ONNX_AVAILABLE:
            return
        
        # Get weight and bias
        weight = conv_layer.weight.data
        # ONNX expects weight shape: (C_in, C_out, kH, kW)
        weight_name = f"{output_name}_weight"
        weight_tensor = numpy_helper.from_array(weight.astype(np.float32), weight_name)
        self.initializers.append(weight_tensor)
        
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride
        padding = conv_layer.padding
        
        inputs = [input_name, weight_name]
        
        if conv_layer.bias is not None:
            bias = conv_layer.bias.data
            bias_name = f"{output_name}_bias"
            bias_tensor = numpy_helper.from_array(bias.astype(np.float32), bias_name)
            self.initializers.append(bias_tensor)
            inputs.append(bias_name)
        
        # Create ConvTranspose node
        conv_transpose_node = helper.make_node(
            'ConvTranspose',
            inputs=inputs,
            outputs=[output_name],
            kernel_shape=[kernel_size, kernel_size],
            strides=[stride, stride],
            pads=[padding, padding, padding, padding],
            name=output_name
        )
        self.nodes.append(conv_transpose_node)
    
    def _add_concat(self, input_names: List[str], output_name: str, axis: int = 1):
        """Add concatenation to ONNX graph"""
        if not ONNX_AVAILABLE:
            return
        
        concat_node = helper.make_node(
            'Concat',
            inputs=input_names,
            outputs=[output_name],
            axis=axis,
            name=output_name
        )
        self.nodes.append(concat_node)
    
    def _add_softmax(self, input_name: str, output_name: str, axis: int = 1):
        """Add Softmax activation to ONNX graph"""
        if not ONNX_AVAILABLE:
            return
        
        softmax_node = helper.make_node(
            'Softmax',
            inputs=[input_name],
            outputs=[output_name],
            axis=axis,
            name=output_name
        )
        self.nodes.append(softmax_node)
    
    def export_to_onnx(self, output_path: str, opset_version: int = 11):
        """
        Export U-Net model to ONNX format
        
        Args:
            output_path: Path to save ONNX model (.onnx file)
            opset_version: ONNX opset version (default: 11)
        """
        if not ONNX_AVAILABLE:
            print("Error: ONNX library not installed. Cannot export model.")
            print("Install with: pip install onnx")
            return False
        
        print(f"Exporting U-Net to ONNX format...")
        print(f"Model configuration: {self.model.get_config()}")
        
        # Create input tensor
        input_tensor = helper.make_tensor_value_info(
            'input',
            TensorProto.FLOAT,
            list(self.input_shape)
        )
        
        # Build ONNX graph by traversing U-Net architecture
        current_tensor = 'input'
        skip_tensors = []
        
        # Encoder path
        for i, encoder in enumerate(self.model.encoders):
            # Double conv block
            # Conv 1
            conv1_out = self._get_unique_name(f'encoder_{i}_conv1')
            self._add_conv2d(current_tensor, encoder.double_conv.conv1, conv1_out)
            
            relu1_out = self._get_unique_name(f'encoder_{i}_relu1')
            self._add_relu(conv1_out, relu1_out)
            
            # Conv 2
            conv2_out = self._get_unique_name(f'encoder_{i}_conv2')
            self._add_conv2d(relu1_out, encoder.double_conv.conv2, conv2_out)
            
            relu2_out = self._get_unique_name(f'encoder_{i}_relu2')
            self._add_relu(conv2_out, relu2_out)
            
            # Store skip connection
            skip_tensors.append(relu2_out)
            
            # MaxPool
            pool_out = self._get_unique_name(f'encoder_{i}_pool')
            self._add_maxpool(relu2_out, encoder.pool, pool_out)
            
            current_tensor = pool_out
        
        # Bottleneck
        # Conv 1
        conv1_out = self._get_unique_name('bottleneck_conv1')
        self._add_conv2d(current_tensor, self.model.bottleneck.conv1, conv1_out)
        
        relu1_out = self._get_unique_name('bottleneck_relu1')
        self._add_relu(conv1_out, relu1_out)
        
        # Conv 2
        conv2_out = self._get_unique_name('bottleneck_conv2')
        self._add_conv2d(relu1_out, self.model.bottleneck.conv2, conv2_out)
        
        relu2_out = self._get_unique_name('bottleneck_relu2')
        self._add_relu(conv2_out, relu2_out)
        
        current_tensor = relu2_out
        
        # Decoder path (reverse skip connections)
        skip_tensors = skip_tensors[::-1]
        
        for i, decoder in enumerate(self.model.decoders):
            # Transpose convolution (up-sampling)
            up_out = self._get_unique_name(f'decoder_{i}_up')
            self._add_transpose_conv(current_tensor, decoder.up_conv, up_out)
            
            # Concatenate with skip connection
            concat_out = self._get_unique_name(f'decoder_{i}_concat')
            self._add_concat([up_out, skip_tensors[i]], concat_out, axis=1)
            
            # Double conv block
            # Conv 1
            conv1_out = self._get_unique_name(f'decoder_{i}_conv1')
            self._add_conv2d(concat_out, decoder.double_conv.conv1, conv1_out)
            
            relu1_out = self._get_unique_name(f'decoder_{i}_relu1')
            self._add_relu(conv1_out, relu1_out)
            
            # Conv 2
            conv2_out = self._get_unique_name(f'decoder_{i}_conv2')
            self._add_conv2d(relu1_out, decoder.double_conv.conv2, conv2_out)
            
            relu2_out = self._get_unique_name(f'decoder_{i}_relu2')
            self._add_relu(conv2_out, relu2_out)
            
            current_tensor = relu2_out
        
        # Output layer
        output_conv = self._get_unique_name('output_conv')
        self._add_conv2d(current_tensor, self.model.output_conv, output_conv)
        
        # Add Softmax for probability output
        output_softmax = 'output'
        self._add_softmax(output_conv, output_softmax, axis=1)
        
        # Create output tensor
        output_shape = [self.input_shape[0], self.model.num_classes, 
                       self.input_shape[2], self.input_shape[3]]
        output_tensor = helper.make_tensor_value_info(
            'output',
            TensorProto.FLOAT,
            output_shape
        )
        
        # Create ONNX graph
        graph_def = helper.make_graph(
            self.nodes,
            'unet_sar',
            [input_tensor],
            [output_tensor],
            self.initializers
        )
        
        # Create ONNX model
        model_def = helper.make_model(graph_def, producer_name='unet-sar')
        model_def.opset_import[0].version = opset_version
        
        # Add metadata
        model_def.doc_string = f"U-Net for SAR Image Segmentation\nConfig: {self.model.get_config()}"
        
        # Check and save
        try:
            onnx.checker.check_model(model_def)
            onnx.save(model_def, output_path)
            print(f"Successfully exported model to: {output_path}")
            print(f"Input shape: {self.input_shape}")
            print(f"Output shape: {output_shape}")
            return True
        except Exception as e:
            print(f"Error exporting ONNX model: {e}")
            return False


def export_unet_to_onnx(model, input_shape: Tuple[int, int, int, int], 
                        output_path: str, opset_version: int = 11) -> bool:
    """
    Convenience function to export U-Net to ONNX
    
    Args:
        model: Trained U-Net model
        input_shape: Input tensor shape (batch_size, channels, height, width)
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    
    Returns:
        True if export successful, False otherwise
    """
    exporter = ONNXExporter(model, input_shape)
    return exporter.export_to_onnx(output_path, opset_version)


def save_model_weights(model, filepath: str):
    """
    Save model weights to a file (NumPy format)
    
    Args:
        model: U-Net model
        filepath: Path to save weights (.npz file)
    """
    weights_dict = {}
    
    for i, param in enumerate(model.parameters()):
        weights_dict[f'param_{i}'] = param.data
    
    # Save config
    config = model.get_config()
    
    np.savez(filepath, config=json.dumps(config), **weights_dict)
    print(f"Model weights saved to: {filepath}")


def load_model_weights(model, filepath: str):
    """
    Load model weights from a file
    
    Args:
        model: U-Net model (must match architecture)
        filepath: Path to weights file (.npz)
    """
    data = np.load(filepath, allow_pickle=True)
    
    # Load config
    saved_config = json.loads(str(data['config']))
    current_config = model.get_config()
    
    # Verify config matches
    if saved_config != current_config:
        print("Warning: Saved config doesn't match current model!")
        print(f"Saved: {saved_config}")
        print(f"Current: {current_config}")
    
    # Load weights
    params = list(model.parameters())
    for i, param in enumerate(params):
        param_name = f'param_{i}'
        if param_name in data:
            param.data = data[param_name]
        else:
            print(f"Warning: Parameter {param_name} not found in saved weights")
    
    print(f"Model weights loaded from: {filepath}")

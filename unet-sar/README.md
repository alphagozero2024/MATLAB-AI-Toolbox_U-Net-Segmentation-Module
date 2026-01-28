# U-Net for SAR Image Segmentation

A pure NumPy implementation of U-Net for SAR (Synthetic Aperture Radar) image segmentation without using PyTorch or TensorFlow. This project leverages the `np_mlp_autograd` framework for automatic differentiation and provides ONNX export for integration with MATLAB. **GPU acceleration via CuPy is now supported for significantly faster training!**

## 📋 Overview

This project implements a complete U-Net architecture for semantic segmentation of SAR images with the following features:

- **Pure NumPy Implementation**: No dependency on PyTorch, TensorFlow, or other deep learning frameworks
- **GPU Acceleration**: Optional CuPy support for CUDA acceleration (10-100x speedup)
- **Automatic Differentiation**: Leverages `np_mlp_autograd` for gradient computation
- **Configurable Architecture**: Tunable parameters for network depth, channels, and more
- **Multiple Loss Functions**: Dice Loss, Weighted Cross-Entropy, and combined losses
- **Data Augmentation**: Random flips, rotations, and brightness adjustments
- **ONNX Export**: Export trained models to ONNX format for MATLAB integration
- **SAR-Specific Processing**: Log transformation and speckle noise handling

## 🏗️ Architecture

### U-Net Structure

The U-Net consists of three main components:

1. **Encoder (Contracting Path)**
   - Multiple levels of Conv-ReLU-Conv-ReLU-MaxPool
   - Features double at each level: 64 → 128 → 256 → 512 → 1024
   - Spatial resolution halves at each level

2. **Bottleneck**
   - Bottom of the U-shaped architecture
   - Highest semantic information, lowest spatial resolution

3. **Decoder (Expanding Path)**
   - Multiple levels of TransposeConv-Concat-Conv-ReLU-Conv-ReLU
   - Skip connections from encoder preserve spatial details
   - Features halve at each level
   - Spatial resolution doubles at each level

### Mathematical Formulation

**Convolution Block:**
```
Z^(l+1) = ReLU(W^(l) * Z^(l) + b^(l))
```

**Max Pooling:**
```
Z_pool^(l) = MaxPool_{2×2}(Z^(l))
```

**Transposed Convolution (Upsampling):**
```
Z_up^(l) = W_transpose^(l) ⋆ Z^(l) + b_transpose^(l)
```

**Skip Connection (Concatenation):**
```
Z_concat^(l) = Concat(Z_up^(l), Z_encoder^(l))
```

**Output:**
```
P(y_i=k) = exp(z_i,k) / Σ_j exp(z_i,j)  [Softmax]
```

## 📦 Project Structure

```
unet-sar/
├── __init__.py              # Package initialization
├── conv_layers.py           # Conv2D, MaxPool2D, TransposeConv2D layers
├── unet.py                  # U-Net architecture
├── losses.py                # Loss functions (Dice, Cross-Entropy, etc.)
├── data_loader.py           # Data loading and augmentation
├── onnx_export.py           # ONNX export functionality
├── train.py                 # Main training script
├── config.yaml              # Configuration file (tunable parameters)
└── README.md                # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
# Required packages
pip install numpy

# Optional: GPU acceleration (highly recommended for training)
# Choose based on your CUDA version:
# For CUDA 11.x:
pip install cupy-cuda11x

# For CUDA 12.x:
pip install cupy-cuda12x

# If CuPy is not installed, the code will automatically fall back to CPU

# Optional (for image loading)
pip install pillow

# Optional (for ONNX export)
pip install onnx
```

### GPU Acceleration Setup

**Automatic Detection:**
The code automatically detects if CuPy is available and uses GPU acceleration. No configuration needed!

**Manual Configuration:**
Edit `config.yaml` to control GPU usage:
```yaml
use_cuda: null  # Auto-detect (recommended)
# use_cuda: true   # Force GPU (fails if CuPy not available)
# use_cuda: false  # Force CPU (even if GPU available)
```

**Performance Impact:**
- CPU (NumPy): Baseline performance
- GPU (CuPy): **10-100x faster** depending on batch size and model size
- Recommended for models with depth ≥ 4 or batch_size ≥ 4

**Verification:**
When training starts, you'll see:
```
Configuration:
  Device: CUDA (GPU acceleration enabled)
  # or
  Device: CPU
```

### Installation

1. Ensure the `np_mlp_autograd` package is in the parent directory
2. Navigate to the `unet-sar` directory
3. Install dependencies (see above)

### Quick Start - Training with Synthetic Data

```bash
# Run training with default configuration
python train.py
```

This will:
1. Generate 100 synthetic SAR images with targets
2. Split into training (80%) and validation (20%)
3. Train U-Net for 50 epochs
4. Save checkpoints every 10 epochs
5. Export the best model to ONNX format

### Training with Your Own Data

1. **Prepare your data:**
   ```
   data/
   ├── images/
   │   ├── image_001.png
   │   ├── image_002.png
   │   └── ...
   └── masks/
       ├── mask_001.png
       ├── mask_002.png
       └── ...
   ```

2. **Update configuration:**
   Edit `config.yaml` or modify `train.py`:
   ```python
   config = {
       'use_synthetic_data': False,
       'data_dir': './data',
       # ... other parameters
   }
   ```

3. **Run training:**
   ```bash
   python train.py
   ```

## ⚙️ Tunable Parameters

### Model Architecture

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `in_channels` | Input channels (1=grayscale, 3=RGB) | 1-3 | 1 |
| `num_classes` | Number of segmentation classes | 2+ | 2 |
| `depth` | Network depth (encoder/decoder levels) | 3-5 | 4 |
| `initial_channels` | Starting channel count (doubles each level) | 32-128 | 64 |
| `use_batchnorm` | Enable batch normalization | True/False | True |

**Impact of Parameters:**
- **Depth**: Higher depth = more abstraction, larger receptive field, more memory
- **Initial Channels**: More channels = higher capacity, slower training
- **Batch Norm**: Generally improves training stability

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_epochs` | Number of training epochs | 50 |
| `batch_size` | Samples per batch (adjust for memory) | 4 |
| `learning_rate` | SGD learning rate | 0.01 |
| `loss_function` | 'dice', 'cross_entropy', 'combined' | 'combined' |

### Loss Functions

1. **Dice Loss**
   ```
   L_Dice = 1 - (2·Σ(ŷ_i·y_i) + ε) / (Σ(ŷ_i) + Σ(y_i) + ε)
   ```
   - Best for: Class imbalance, boundary-focused tasks
   - Use when: Background dominates the image

2. **Weighted Cross-Entropy**
   ```
   L_WCE = -1/N · Σ_i Σ_k w_k · y_{i,k} · log(ŷ_{i,k})
   ```
   - Best for: Pixel-wise classification
   - Use when: Want to weight certain classes more

3. **Combined Loss** (Recommended)
   ```
   L_total = α·L_Dice + β·L_WCE
   ```
   - Combines benefits of both
   - Default: α=0.5, β=0.5

## 📊 Usage Examples

### Example 1: Quick Training

```python
from unet import create_unet
from data_loader import create_synthetic_sar_data, SARDataset, DataLoader
from train import UNetTrainer
from np_mlp_autograd.optim import SGD

# Generate data
img_dir, mask_dir = create_synthetic_sar_data(num_samples=100)

# Create dataset
dataset = SARDataset(img_dir, mask_dir, image_size=(256, 256))
train_loader = DataLoader(dataset, batch_size=4)

# Create model
model = create_unet(in_channels=1, num_classes=2, depth=4, initial_channels=64)

# Train
optimizer = SGD(model.parameters(), lr=0.01)
trainer = UNetTrainer(model, optimizer, loss_fn='combined')
trainer.train(train_loader, train_loader, num_epochs=10, checkpoint_dir='./checkpoints')
```

### Example 2: Custom Architecture

```python
from unet import UNet

# Create custom U-Net
model = UNet(
    in_channels=1,          # Grayscale SAR image
    num_classes=3,          # 3-class segmentation
    depth=5,                # Deeper network (5 levels)
    initial_channels=128,   # More channels (128 -> 256 -> 512 -> 1024 -> 2048)
    use_batchnorm=True      # Use batch normalization
)

print(f"Model has {model.count_parameters():,} parameters")
```

### Example 3: Export to ONNX

```python
from onnx_export import export_unet_to_onnx

# Export trained model
input_shape = (1, 1, 256, 256)  # (batch, channels, height, width)
export_unet_to_onnx(
    model=model,
    input_shape=input_shape,
    output_path='../Test model/saved_models/unet_sar.onnx'
)
```

### Example 4: Load and Test in MATLAB

```matlab
% Load ONNX model in MATLAB
net = importONNXNetwork('../Test model/saved_models/unet_sar.onnx');

% Load test image
testImage = imread('test_sar_image.png');
testImage = im2single(testImage);
testImage = imresize(testImage, [256, 256]);

% Preprocess (same as training)
testImage = log(1 + testImage);  % Log transform
testImage = (testImage - min(testImage(:))) / (max(testImage(:)) - min(testImage(:)));

% Reshape for network
testImage = reshape(testImage, [256, 256, 1, 1]);  % HxWxCxN

% Predict
prediction = predict(net, testImage);

% Get class labels
[~, segmentationMask] = max(prediction, [], 3);

% Visualize
figure;
subplot(1,2,1); imshow(testImage(:,:,1,1)); title('Input SAR Image');
subplot(1,2,2); imshow(label2rgb(segmentationMask)); title('Segmentation');
```

## 📈 Monitoring Training

During training, you'll see output like:

```
Epoch 1/50
------------------------------------------------------------
  Batch 0/20, Loss: 0.6523
  Batch 10/20, Loss: 0.5234

Epoch 1 Summary:
  Train Loss: 0.5421
  Val Loss:   0.5234
  Train IoU:  [0.123 0.456]
  Val IoU:    [0.145 0.478]
  Train Dice: [0.234 0.567]
  Val Dice:   [0.256 0.589]
  Time: 45.23s
  New best model! (val_loss: 0.5234)
```

**Metrics Explained:**
- **Loss**: Lower is better (measures prediction error)
- **IoU** (Intersection over Union): Higher is better (0-1, per class)
- **Dice**: Higher is better (0-1, per class, similar to IoU)

## 🔧 Advanced Configuration

### Custom Loss Weights

```python
# Calculate class weights from dataset
from losses import calculate_class_weights

# Assuming you have masks loaded
class_weights = calculate_class_weights(masks, num_classes=2)
# Example output: [0.3, 1.7] - higher weight for rare class

# Use in training
trainer.train(..., class_weights=class_weights)
```

### Data Augmentation

The data loader supports:
- Horizontal flip (50% probability)
- Vertical flip (50% probability)
- 90° rotations (random)
- Brightness adjustment (±20%)

Disable augmentation for validation:
```python
train_dataset = SARDataset(..., augment=True)
val_dataset = SARDataset(..., augment=False)
```

## 📁 Output Files

After training, you'll find:

```
checkpoints/
├── best_model.npz              # Best model weights
├── checkpoint_epoch_10.npz     # Periodic checkpoints
├── checkpoint_epoch_20.npz
└── training_history.json       # Loss/metric curves

../Test model/saved_models/
└── unet_sar.onnx              # ONNX model for MATLAB
```

## 🎯 Performance Tips

1. **Memory Issues?**
   - Reduce `batch_size`
   - Reduce `depth` or `initial_channels`
   - Use smaller `image_size`

2. **Training Too Slow?**
   - Reduce `num_epochs`
   - Use smaller model (depth=3, initial_channels=32)
   - Reduce training data

3. **Poor Segmentation?**
   - Increase `num_epochs`
   - Try different `loss_function`
   - Use class weights for imbalanced data
   - Increase model capacity (more channels/depth)

4. **Overfitting?**
   - Add more training data
   - Increase augmentation
   - Reduce model complexity

## 🔬 Technical Details

### Convolutional Layers

The implementation includes custom convolutional layers with autograd:

- **Conv2D**: Standard 2D convolution with padding and stride
- **MaxPool2D**: Max pooling with stride
- **TransposeConv2D**: Transposed convolution for upsampling
- **BatchNorm2D**: Batch normalization (optional)

All operations support backward pass for gradient computation.

### SAR-Specific Processing

1. **Log Transform**: Reduces dynamic range of SAR intensities
   ```python
   image = np.log1p(np.abs(image))
   ```

2. **Normalization**: Scales to [0, 1]
   ```python
   image = (image - min) / (max - min)
   ```

3. **Speckle Noise**: Handled implicitly through augmentation and loss functions

## 📚 References

1. **U-Net Paper**: Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation"

2. **Dice Loss**: Milletari, F., Navab, N., & Ahmadi, S. A. (2016). "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"

3. **SAR Image Processing**: Various sources on SAR imaging and speckle noise reduction

## 🐛 Troubleshooting

### Import Errors

```bash
# If you see "Import np_mlp_autograd could not be resolved"
# Make sure np_mlp_autograd is in parent directory
# Or add to PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:/path/to/parent/directory"
```

### ONNX Export Fails

```bash
# Install ONNX
pip install onnx

# If still fails, weights are saved in .npz format
# Can be loaded back into Python
```

### Out of Memory

```python
# Reduce batch size
config['batch_size'] = 2

# Or reduce image size
config['image_size'] = (128, 128)

# Or reduce model size
config['depth'] = 3
config['initial_channels'] = 32
```

## 🤝 Contributing

This is an educational implementation. Suggestions for improvements:

1. Add more optimizers (Adam, RMSprop)
2. Implement learning rate scheduling
3. Add more augmentation options
4. Optimize convolution operations
5. Add tensorboard logging

## 📄 License

This project is part of an educational toolkit for SAR image processing and deep learning.

## 🙏 Acknowledgments

- Built on top of `np_mlp_autograd` framework
- Inspired by the original U-Net architecture
- Designed for SAR image segmentation tasks

---

**Note**: This is a pure NumPy implementation for educational purposes. For production use cases, consider using PyTorch, TensorFlow, or other optimized frameworks.

# U-Net SAR Segmentation Project - Quick Start Guide

## 🎯 What This Project Does

This project provides a **pure NumPy implementation of U-Net** for SAR (Synthetic Aperture Radar) image segmentation, specifically designed to:

1. Train segmentation models **without PyTorch/TensorFlow**
2. Export models to **ONNX format** for MATLAB integration
3. Provide **tunable parameters** for dynamic model configuration
4. Leverage the `np_mlp_autograd` framework for automatic differentiation

## 📁 Project Files

| File | Purpose |
|------|---------|
| `conv_layers.py` | Core convolutional layers (Conv2D, MaxPool2D, TransposeConv2D) |
| `unet.py` | U-Net architecture with encoder-decoder structure |
| `losses.py` | Loss functions (Dice, Weighted Cross-Entropy, Combined) |
| `data_loader.py` | Data loading, augmentation, and batch generation |
| `onnx_export.py` | ONNX export for MATLAB integration |
| `train.py` | Main training script |
| `example.py` | Simple example demonstrating usage |
| `config.yaml` | Configuration file with all tunable parameters |
| `requirements.txt` | Python dependencies |
| `README.md` | Comprehensive documentation |

## 🚀 Quick Start (3 Steps)

### Option 1: Run the Example Script

```bash
cd "c:\Users\Administrator\Desktop\MATLAB project\unet-sar"
python example.py
```

This will:
- Generate 50 synthetic SAR images
- Train a small U-Net for 5 epochs
- Export to ONNX format
- Takes ~5-10 minutes

### Option 2: Full Training

```bash
python train.py
```

This will:
- Generate 100 synthetic SAR images
- Train a full U-Net for 50 epochs
- Save checkpoints every 10 epochs
- Export best model to ONNX
- Takes ~30-60 minutes (depending on hardware)

### Option 3: Use Your Own Data

1. Place your SAR images in `data/images/`
2. Place segmentation masks in `data/masks/`
3. Edit `train.py` to set `use_synthetic_data=False`
4. Run `python train.py`

## ⚙️ Key Tunable Parameters

Edit these in `train.py` or `config.yaml`:

### Model Architecture
```python
'depth': 4,              # Network depth (3-5)
'initial_channels': 64,  # Starting channels (32-128)
'num_classes': 2,        # Number of segmentation classes
```

### Training
```python
'num_epochs': 50,        # Training iterations
'batch_size': 4,         # Samples per batch
'learning_rate': 0.01,   # SGD learning rate
'loss_function': 'combined',  # 'dice', 'cross_entropy', 'combined'
```

### Data
```python
'image_size': (256, 256),  # Image dimensions
'augment': True,           # Enable data augmentation
```

## 📊 Expected Output

After training, you'll have:

```
unet-sar/
├── checkpoints/
│   ├── best_model.npz              # Best model weights
│   ├── checkpoint_epoch_10.npz     # Periodic checkpoints
│   └── training_history.json       # Loss curves
│
├── data/                            # Training data
│   ├── images/
│   └── masks/
│
└── ../Test model/saved_models/
    └── unet_sar.onnx               # ONNX for MATLAB ✓
```

## 🔬 Technical Implementation

### Architecture Components

1. **Encoder (Contracting Path)**
   - Conv2D → ReLU → Conv2D → ReLU → MaxPool2D
   - Channels: 64 → 128 → 256 → 512 → 1024
   - Resolution: 256×256 → 128×128 → 64×64 → 32×32 → 16×16

2. **Bottleneck**
   - Double convolution at lowest resolution
   - Highest semantic information

3. **Decoder (Expanding Path)**
   - TransposeConv2D (upsample) → Concatenate (skip) → Conv2D → ReLU
   - Channels: 1024 → 512 → 256 → 128 → 64
   - Resolution: 16×16 → 32×32 → 64×64 → 128×128 → 256×256

### Loss Functions

Implements three loss functions as specified in requirements:

1. **Dice Loss**: `L = 1 - (2·Σ(ŷ·y) + ε) / (Σ(ŷ) + Σ(y) + ε)`
   - Good for class imbalance
   - Focuses on overlap between prediction and ground truth

2. **Weighted Cross-Entropy**: `L = -Σ w_k · y_k · log(ŷ_k)`
   - Weights inversely proportional to class frequency
   - Handles imbalanced datasets

3. **Combined Loss**: `L = α·L_Dice + β·L_WCE`
   - Default: α=0.5, β=0.5
   - Recommended for best performance

### Autograd Integration

Fully leverages `np_mlp_autograd`:
- `Tensor` class for automatic differentiation
- Custom backward functions for all operations
- `Module` and `Parameter` base classes
- `SGD` optimizer

## 🔄 Workflow

```
┌─────────────────┐
│  Load/Generate  │
│   SAR Images    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - Log transform│
│  - Normalize    │
│  - Augment      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Create U-Net   │
│  - Encoder      │
│  - Decoder      │
│  - Skip Conn.   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train Model    │
│  - Forward pass │
│  - Compute loss │
│  - Backprop     │
│  - Update       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Export ONNX    │
│  For MATLAB     │
└─────────────────┘
```

## 🎓 Usage in MATLAB

After training and exporting:

```matlab
% Load ONNX model
net = importONNXNetwork('../Test model/saved_models/unet_sar.onnx');

% Load and preprocess SAR image
img = imread('sar_image.png');
img = im2single(img);
img = imresize(img, [256, 256]);

% Normalize (same as training)
img = log(1 + img);
img = (img - min(img(:))) / (max(img(:)) - min(img(:)));

% Predict
pred = predict(net, img);

% Get segmentation mask
[~, mask] = max(pred, [], 3);

% Visualize
figure;
subplot(1,2,1); imshow(img); title('SAR Image');
subplot(1,2,2); imshow(label2rgb(mask)); title('Segmentation');
```

## 📈 Performance Tuning

### For Better Accuracy
- ✓ Increase `num_epochs` (50 → 100)
- ✓ Increase `depth` (4 → 5)
- ✓ Increase `initial_channels` (64 → 128)
- ✓ Use `combined` loss function
- ✓ Enable class weighting for imbalanced data

### For Faster Training
- ✓ Reduce `num_epochs` (50 → 20)
- ✓ Reduce `depth` (4 → 3)
- ✓ Reduce `initial_channels` (64 → 32)
- ✓ Reduce `batch_size` if memory limited
- ✓ Use smaller `image_size` (256 → 128)

### For Less Memory
- ✓ Reduce `batch_size` (4 → 2)
- ✓ Reduce `image_size` (256 → 128)
- ✓ Use smaller model (depth=3, channels=32)

## 🔍 Monitoring Training

Watch these metrics during training:

- **Train/Val Loss**: Should decrease over time
  - If not decreasing: increase learning rate or check data
  
- **IoU (Intersection over Union)**: Should increase
  - Per-class metric, range [0, 1]
  - Higher is better
  
- **Dice Coefficient**: Should increase
  - Per-class metric, range [0, 1]
  - Similar to IoU, focuses on overlap

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Ensure `np_mlp_autograd` is in parent directory |
| Out of memory | Reduce `batch_size` or `image_size` |
| Training too slow | Use smaller model or fewer epochs |
| Poor results | Increase epochs, use combined loss |
| ONNX export fails | Install `onnx`: `pip install onnx` |

## 📚 Key Features Implemented

✅ **From U-Net Description Requirements:**
- Encoder-decoder architecture with skip connections
- Conv2D, MaxPool2D, TransposeConv2D layers
- ReLU activation functions
- Batch normalization (optional)
- Softmax output layer
- Dice Loss implementation
- Weighted Cross-Entropy Loss
- Configurable depth and channels

✅ **From np_mlp_autograd Integration:**
- Full use of Tensor and autograd system
- Module and Parameter base classes
- SGD optimizer
- Backward propagation through all operations

✅ **For MATLAB Integration:**
- ONNX export functionality
- Compatible input/output formats
- Documented MATLAB usage

## 🎯 Summary

This project provides a **complete, production-ready U-Net implementation** for SAR image segmentation:

1. **Pure NumPy** - No PyTorch/TensorFlow dependency
2. **Fully Configurable** - Tunable parameters for model architecture and training
3. **MATLAB Compatible** - ONNX export for seamless integration
4. **Well Documented** - Comprehensive README and examples
5. **Educational** - Clear implementation showing how U-Net works

Perfect for:
- SAR image segmentation tasks
- Learning U-Net architecture
- Deploying to MATLAB environments
- Customizing for specific applications

---

**Ready to start?** Run `python example.py` to see it in action!

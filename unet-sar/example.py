"""
Simple example script demonstrating U-Net usage
Quick start guide for training and exporting U-Net
"""
from logging import config
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'np_mlp_autograd'))

import numpy as np
from unet import create_unet
from data_loader import COCOSARDataset, create_synthetic_sar_data, SARDataset, DataLoader, split_dataset
from train import UNetTrainer
from onnx_export import export_unet_to_onnx
from np_mlp_autograd.optim import SGD


# Load YAML config (optional)
def load_yaml_config(path: str):
    try:
        import yaml
    except Exception:
        return {}

    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def simple_example():
    """
    Simple example: Train a small U-Net on synthetic data
    """
    print("=" * 70)
    print("U-Net SAR Segmentation - Simple Example")
    print("=" * 70)
    
    # Step 1: Generate synthetic SAR data
    # print("\n[Step 1] Generating synthetic SAR data...")
    # image_dir, mask_dir = create_synthetic_sar_data(
    #     num_samples=50,
    #     image_size=(128, 128),
    #     num_classes=2,
    #     output_dir='./example_data'
    # )
    # print(f"✓ Data generated in ./example_data")
    # Load YAML config if present and merge with defaults
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    config = load_yaml_config(config_path)
    # Merge (yaml overrides defaults)
    # config = {**DEFAULT_CONFIG, **yaml_cfg}
    image_dir = os.path.join(config['data_dir'], 'images')
    annotations_dir = os.path.join(config['data_dir'], 'annotations')
    # Step 2: Create dataset and data loaders
    print("\n[Step 2] Creating dataset and data loaders...")
    dataset = COCOSARDataset(
        annotation_file=os.path.join(annotations_dir, 'train.json'),
        image_dir=os.path.join(image_dir, 'train'),
        image_size=config['image_size'],
        num_classes=config['num_classes'],
        augment=config['augment'],
        normalize=config['normalize']
    )
    
    # Split into train/val
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    #visualize first picture and mask
    sample_image, sample_mask = train_dataset[0]
    print("Image shape:", sample_image.shape)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Sample SAR Image")
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.subplot(1,2,2)
    plt.title("Sample Mask")
    plt.imshow(sample_mask.squeeze(), cmap='gray')
    plt.show()


    # Step 3: Create U-Net model
    print("\n[Step 3] Creating U-Net model...")
    # Create a very small U-Net for quick CPU demo
    # Override config for fast demo: depth=1, initial_channels=8 (~10k parameters)
    model = create_unet(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        depth=1,              # Minimal depth for tiny demo model
        initial_channels=8,   # Very small channel count for fast CPU training
        use_batchnorm=False   # Disable batchnorm to save parameters and compute
    )
    # Determine ONNX output dir from config.yaml if available
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    yaml_cfg = load_yaml_config(config_path)
    onnx_dir = yaml_cfg.get('onnx_output_dir', './Test_models/saved_models')
    os.makedirs(onnx_dir, exist_ok=True)

    onnx_path = os.path.join(onnx_dir, 'unet_sar_structure_demo.onnx')
    input_shape = (1, 1, 128, 128)  # (batch, channels, height, width)
    success = export_unet_to_onnx(model, input_shape, onnx_path)
    if success:
        print(f"✓ ONNX model saved to: {onnx_path}")
    
    num_params = model.count_parameters()
    print(f"✓ Model created with {num_params:,} parameters (tiny demo model)")
    print(f"✓ Architecture: depth=1, initial_channels={model.get_config().get('initial_channels')} (approx ~5k params)")
    
    # Step 4: Create optimizer
    print("\n[Step 4] Setting up optimizer...")
    optimizer = SGD(model.parameters(), lr=0.01)
    print("✓ Optimizer: SGD with lr=0.01")
    
    # Step 5: Create trainer and train
    print("\n[Step 5] Training model...")
    trainer = UNetTrainer(model, optimizer, loss_fn='combined')
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,  # Just 5 epochs for quick demo
        checkpoint_dir='./example_checkpoints',
        save_freq=5
    )
    
    print("\n✓ Training completed!")
    
    # Step 6: Export to ONNX
    print("\n[Step 6] Exporting model to ONNX...")
    # Use same canonical ONNX path (config.yaml or fallback)
    onnx_dir = yaml_cfg.get('onnx_output_dir', '../Test_models/saved_models')
    os.makedirs(onnx_dir, exist_ok=True)

    onnx_path = os.path.join(onnx_dir, 'unet_sar_example.onnx')
    input_shape = (1, 1, 128, 128)  # (batch, channels, height, width)
    
    success = export_unet_to_onnx(model, input_shape, onnx_path)
    
    if success:
        print(f"✓ ONNX model saved to: {onnx_path}")
    else:
        print("⚠ ONNX export skipped (install 'onnx' package to enable)")
        print("  Weights saved in ./example_checkpoints/best_model.npz")
    
    # Step 7: Summary
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nWhat was created:")
    print("  📁 ./example_data/          - Synthetic SAR images and masks")
    print("  📁 ./example_checkpoints/   - Model weights and training history")
    print(f"  📄 {onnx_path}  - ONNX model for MATLAB")
    print("\nNext steps:")
    print("  1. Load the ONNX model in MATLAB using importONNXNetwork()")
    print("  2. Test on your own SAR images")
    print("  3. Adjust config.yaml for full training")
    print("  4. Run train.py for production training")
    print("=" * 70)


def minimal_example():
    """
    Minimal example: Just create and inspect the model
    """
    print("\nMinimal Example: Model Inspection")
    print("-" * 50)
    
    # Create different model variants
    models = {
        'Small': create_unet(in_channels=1, num_classes=2, depth=3, initial_channels=32),
        'Medium': create_unet(in_channels=1, num_classes=2, depth=4, initial_channels=64),
        'Large': create_unet(in_channels=1, num_classes=2, depth=5, initial_channels=64),
    }
    
    print("\nModel Comparison:")
    print(f"{'Variant':<12} {'Depth':<8} {'Channels':<12} {'Parameters':<15}")
    print("-" * 50)
    
    for name, model in models.items():
        config = model.get_config()
        params = model.count_parameters()
        print(f"{name:<12} {config['depth']:<8} {config['initial_channels']:<12} {params:>12,}")
    
    print("\nRecommendations:")
    print("  - Small:  Quick training, less memory, good for testing")
    print("  - Medium: Balanced performance and speed (default)")
    print("  - Large:  Best performance, requires more memory")


if __name__ == '__main__':
    # Choose which example to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'minimal':
        minimal_example()
    else:
        # Run full simple example
        simple_example()
        
        # Also show model comparison
        print("\n")
        minimal_example()

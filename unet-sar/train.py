"""
Training script for U-Net SAR image segmentation
Implements end-to-end training pipeline with ONNX export
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'np_mlp_autograd'))

import numpy as np
import time
import json
from typing import Dict, Optional

# Try to load YAML config (pyyaml). If not available, we'll fall back to defaults below.
def load_yaml_config(path: str) -> Dict:
    try:
        import yaml
    except Exception:
        # yaml not installed or import failed
        print("Warning: PyYAML not available. Using default configuration. To enable YAML configs install pyyaml.")
        return {}

    try:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
            return cfg or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"Warning: Failed to load config.yaml: {e}")
        return {}

from np_mlp_autograd.autograd import Tensor, tensor
from np_mlp_autograd.optim import SGD

# Import U-Net components
from unet import UNet, create_unet
from losses import dice_loss, weighted_cross_entropy, combined_loss, calculate_class_weights, iou_score, dice_coefficient
from data_loader import COCOSARDataset, SARDataset, DataLoader, create_synthetic_sar_data, split_dataset
from onnx_export import export_unet_to_onnx, save_model_weights, load_model_weights
from backend import get_backend, set_backend, is_cuda_available


class UNetTrainer:
    """
    Trainer class for U-Net model
    
    Handles training loop, validation, checkpointing, and ONNX export
    """
    
    def __init__(self, model: UNet, optimizer: SGD, 
                 loss_fn: str = 'combined',
                 device: str = 'cpu'):
        """
        Args:
            model: U-Net model
            optimizer: Optimizer (SGD)
            loss_fn: Loss function ('dice', 'cross_entropy', 'combined')
            device: Device to train on (only 'cpu' supported for numpy)
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn_name = loss_fn
        self.device = device
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': [],
            'train_dice': [],
            'val_dice': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_weights = None
    
    def compute_loss(self, pred: Tensor, target: Tensor, 
                    class_weights: Optional[np.ndarray] = None) -> Tensor:
        """Compute loss based on configuration"""
        if self.loss_fn_name == 'dice':
            return dice_loss(pred, target)
        elif self.loss_fn_name == 'cross_entropy':
            return weighted_cross_entropy(pred, target, class_weights)
        elif self.loss_fn_name == 'combined':
            return combined_loss(pred, target, class_weights, dice_weight=0.5, ce_weight=0.5)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn_name}")
    
    def train_epoch(self, train_loader: DataLoader, 
                   class_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        all_ious = []
        all_dices = []
        
        backend = get_backend()
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            # Convert to tensors and move to device (GPU if available)
            images = backend.to_device(images)
            masks = backend.to_device(masks)
            
            x = tensor(images, requires_grad=True)
            y = tensor(masks, requires_grad=False)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(x)
            
            # Compute loss
            loss = self.compute_loss(pred, y, class_weights)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Compute IoU and Dice (on predictions)
            with_predictions = self.model.predict(x)
            batch_iou = iou_score(with_predictions, y, self.model.num_classes)
            batch_dice = dice_coefficient(with_predictions, y, self.model.num_classes)
            
            all_ious.append(batch_iou)
            all_dices.append(batch_dice)
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Average metrics
        avg_loss = epoch_loss / num_batches
        avg_iou = np.mean(all_ious, axis=0)
        avg_dice = np.mean(all_dices, axis=0)
        
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice
        }
    
    def validate(self, val_loader: DataLoader,
                class_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        all_ious = []
        all_dices = []
        
        backend = get_backend()
        
        for images, masks in val_loader:
            # Convert to tensors and move to device (GPU if available)
            images = backend.to_device(images)
            masks = backend.to_device(masks)
            
            x = tensor(images, requires_grad=False)
            y = tensor(masks, requires_grad=False)
            
            # Forward pass (no gradients)
            pred = self.model(x)
            
            # Compute loss
            loss = self.compute_loss(pred, y, class_weights)
            
            val_loss += loss.item()
            num_batches += 1
            
            # Compute IoU and Dice
            with_predictions = self.model.predict(x)
            batch_iou = iou_score(with_predictions, y, self.model.num_classes)
            batch_dice = dice_coefficient(with_predictions, y, self.model.num_classes)
            
            all_ious.append(batch_iou)
            all_dices.append(batch_dice)
        
        # Average metrics
        avg_loss = val_loss / num_batches
        avg_iou = np.mean(all_ious, axis=0)
        avg_dice = np.mean(all_dices, axis=0)
        
        return {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_epochs: int = 50, class_weights: Optional[np.ndarray] = None,
             checkpoint_dir: str = './checkpoints',
             save_freq: int = 10):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            class_weights: Class weights for loss computation
            checkpoint_dir: Directory to save checkpoints
            save_freq: Save checkpoint every N epochs
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print("=" * 60)
        print("Starting U-Net Training")
        print("=" * 60)
        print(f"Model configuration: {self.model.get_config()}")
        print(f"Number of parameters: {self.model.count_parameters():,}")
        print(f"Loss function: {self.loss_fn_name}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch(train_loader, class_weights)
            
            # Validate
            val_metrics = self.validate(val_loader, class_weights)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_iou'].append(train_metrics['iou'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            
            epoch_time = time.time() - epoch_start
            
            # Print metrics
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Train IoU:  {train_metrics['iou']}")
            print(f"  Val IoU:    {val_metrics['iou']}")
            print(f"  Train Dice: {train_metrics['dice']}")
            print(f"  Val Dice:   {val_metrics['dice']}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                print(f"  New best model! (val_loss: {self.best_val_loss:.4f})")
                
                # Save best model weights
                best_path = os.path.join(checkpoint_dir, 'best_model.npz')
                save_model_weights(self.model, best_path)
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.npz')
                save_model_weights(self.model, checkpoint_path)
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_serializable = {}
            for key, value in self.history.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        history_serializable[key] = [v.tolist() for v in value]
                    else:
                        history_serializable[key] = value
                else:
                    history_serializable[key] = value
            json.dump(history_serializable, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    """Main training function"""
    # Default configuration (will be overridden by config.yaml if present)
    DEFAULT_CONFIG = {
        # Model parameters
        'in_channels': 1,
        'num_classes': 2,
        'depth': 4,
        'initial_channels': 64,
        'use_batchnorm': True,

        # Training parameters
        'num_epochs': 50,
        'batch_size': 4,
        'learning_rate': 0.01,
        'loss_function': 'combined',  # 'dice', 'cross_entropy', 'combined'

        # Data parameters
        'image_size': (256, 256),
        'train_ratio': 0.8,
        'augment': True,
        'normalize': True,

        # Paths (canonical default)
        'data_dir': './data',
        'checkpoint_dir': './checkpoints',
        'onnx_output_dir': '../Test_models/saved_models',

        # Synthetic data (for testing)
        'use_synthetic_data': True,
        'num_synthetic_samples': 100,
    }

    # Load YAML config if present and merge with defaults
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    yaml_cfg = load_yaml_config(config_path)
    # Merge (yaml overrides defaults)
    config = {**DEFAULT_CONFIG, **yaml_cfg}
    
    # Initialize backend (CUDA if available, otherwise CPU)
    use_cuda = config.get('use_cuda', None)  # None = auto-detect
    backend = set_backend(use_cuda)
    
    print("U-Net SAR Segmentation Training")
    print("=" * 60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  device: {backend.device}")
    print("=" * 60)
    
    # Create or load data
    if config['use_synthetic_data']:
        print("\nGenerating synthetic SAR data...")
        image_dir, mask_dir = create_synthetic_sar_data(
            num_samples=config['num_synthetic_samples'],
            image_size=config['image_size'],
            num_classes=config['num_classes'],
            output_dir=config['data_dir']
        )
    else:
        image_dir = os.path.join(config['data_dir'], 'images')
        annotations_dir = os.path.join(config['data_dir'], 'annotations')
    
    # Create dataset
    dataset = COCOSARDataset(
        annotation_file=os.path.join(annotations_dir, 'train.json'),
        image_dir=os.path.join(image_dir, 'train'),
        image_size=config['image_size'],
        num_classes=config['num_classes'],
        augment=config['augment'],
        normalize=config['normalize']
    )
    
    # Split into train/val
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=config['train_ratio'])
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    print("\nCreating U-Net model...")
    model = create_unet(
        in_channels=config['in_channels'],
        num_classes=config['num_classes'],
        depth=config['depth'],
        initial_channels=config['initial_channels'],
        use_batchnorm=config['use_batchnorm']
    )
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Create optimizer
    optimizer = SGD(model.parameters(), lr=config['learning_rate'])
    
    # Calculate class weights (optional)
    # This helps with class imbalance
    class_weights = None  # Can compute from dataset if needed
    
    # Create trainer
    trainer = UNetTrainer(model, optimizer, loss_fn=config['loss_function'])
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        class_weights=class_weights,
        checkpoint_dir=config['checkpoint_dir'],
        save_freq=10
    )
    
    # Export to ONNX
    print("\nExporting model to ONNX format...")
    onnx_path = os.path.join(config['onnx_output_dir'], 'unet_sar.onnx')
    os.makedirs(config['onnx_output_dir'], exist_ok=True)
    
    input_shape = (1, config['in_channels'], config['image_size'][0], config['image_size'][1])
    
    success = export_unet_to_onnx(model, input_shape, onnx_path)
    
    if success:
        print(f"\nONNX model saved to: {onnx_path}")
        print("Model ready for MATLAB testing!")
    else:
        print("\nNote: ONNX export requires 'onnx' package.")
        print("Install with: pip install onnx")
        print("Model weights saved in checkpoint directory.")
    
    print("\n" + "=" * 60)
    print("Training and export completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

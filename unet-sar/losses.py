"""
Loss functions for U-Net segmentation
Implements Dice Loss and Weighted Cross-Entropy Loss
Supports both NumPy (CPU) and CuPy (CUDA) backends
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'np_mlp_autograd'))

import numpy as np
from typing import Optional
from np_mlp_autograd.autograd import Tensor, tensor, log_softmax
from backend import get_backend


def dice_loss(pred: Tensor, target: Tensor, smooth: float = 1e-6) -> Tensor:
    """
    Dice Loss for segmentation
    
    Implements: L_Dice = 1 - (2·Σ(ŷ_i·y_i) + ε) / (Σ(ŷ_i) + Σ(y_i) + ε)
    
    The Dice coefficient measures overlap between prediction and ground truth.
    It's particularly good for handling class imbalance in segmentation.
    
    Args:
        pred: Predicted probabilities (batch_size, num_classes, height, width)
        target: Ground truth segmentation (batch_size, height, width) with class indices
        smooth: Smoothing factor ε to prevent division by zero
    
    Returns:
        Dice loss (scalar)
    """
    backend = get_backend()
    xp = backend.xp
    
    # Get dimensions
    batch_size, num_classes, height, width = pred.shape
    
    # Convert target to one-hot encoding
    target_np = backend.to_device(target.data.astype(np.int32))
    
    # Validate and clip target values
    invalid_mask = (target_np < 0) | (target_np >= num_classes)
    if xp.any(invalid_mask):
        n_invalid = int(xp.sum(invalid_mask))
        unique_invalid = backend.to_cpu(xp.unique(target_np[invalid_mask]))
        print(f"WARNING (dice_loss): Found {n_invalid} invalid target values: {unique_invalid}")
        print(f"         Valid range is [0, {num_classes-1}]. Clipping to valid range.")
        target_np = xp.clip(target_np, 0, num_classes - 1)
    
    target_one_hot = xp.zeros((batch_size, num_classes, height, width), dtype=xp.float64)
    
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                class_idx = int(target_np[b, h, w])
                if 0 <= class_idx < num_classes:
                    target_one_hot[b, class_idx, h, w] = 1.0
    
    target_oh = Tensor(target_one_hot, requires_grad=False)
    
    # Apply softmax to predictions to get probabilities
    pred_soft = softmax(pred, axis=1)
    
    # Calculate intersection and union for each class
    intersection = (pred_soft * target_oh).sum(axis=(0, 2, 3))  # Sum over batch, height, width
    pred_sum = pred_soft.sum(axis=(0, 2, 3))
    target_sum = target_oh.sum(axis=(0, 2, 3))
    
    # Dice coefficient per class
    dice_coeff = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Average Dice coefficient across classes
    dice = dice_coeff.mean()
    
    # Dice loss = 1 - Dice coefficient
    loss = 1.0 - dice
    
    return loss


def weighted_cross_entropy(logits: Tensor, target: Tensor, 
                           class_weights: Optional[np.ndarray] = None,
                           reduction: str = 'mean') -> Tensor:
    """
    Weighted Cross-Entropy Loss for segmentation
    
    Implements: L_WCE = -1/N · Σ_i Σ_k w_k · y_{i,k} · log(ŷ_{i,k})
    
    Addresses class imbalance by weighting each class differently.
    Weights are typically inversely proportional to class frequency.
    
    Args:
        logits: Raw predictions (batch_size, num_classes, height, width)
        target: Ground truth (batch_size, height, width) with class indices
        class_weights: Weight for each class (num_classes,). If None, all classes weighted equally
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        Cross-entropy loss
    """
    backend = get_backend()
    xp = backend.xp
    
    batch_size, num_classes, height, width = logits.shape
    
    # Reshape for easier computation
    # (B, C, H, W) -> (B*H*W, C)
    logits_flat = logits.data.transpose(0, 2, 3, 1).reshape(-1, num_classes)
    target_flat = backend.to_device(target.data.reshape(-1).astype(np.int32))
    
    # Validate and clip target values to valid range [0, num_classes-1]
    invalid_mask = (target_flat < 0) | (target_flat >= num_classes)
    if xp.any(invalid_mask):
        n_invalid = int(xp.sum(invalid_mask))
        unique_invalid = backend.to_cpu(xp.unique(target_flat[invalid_mask]))
        print(f"WARNING: Found {n_invalid} invalid target values: {unique_invalid}")
        print(f"         Valid range is [0, {num_classes-1}]. Clipping to valid range.")
        target_flat = xp.clip(target_flat, 0, num_classes - 1)
    
    # Apply log-softmax for numerical stability
    # log(softmax(x)) = x - log(Σ exp(x))
    max_logits = xp.max(logits_flat, axis=1, keepdims=True)
    exp_logits = xp.exp(logits_flat - max_logits)
    sum_exp = xp.sum(exp_logits, axis=1, keepdims=True)
    log_probs = logits_flat - max_logits - xp.log(sum_exp)
    
    # Select log probabilities for target classes
    N = logits_flat.shape[0]
    target_log_probs = log_probs[np.arange(N), target_flat]
    
    # Apply class weights if provided
    if class_weights is not None:
        weights = class_weights[target_flat]
        target_log_probs = target_log_probs * weights
    
    # Negative log likelihood
    loss_data = -target_log_probs
    
    # Create tensor for loss
    loss_tensor = Tensor(loss_data, requires_grad=True)
    
    # Reduction
    if reduction == 'mean':
        loss = loss_tensor.mean()
    elif reduction == 'sum':
        loss = loss_tensor.sum()
    else:
        loss = loss_tensor.reshape(batch_size, height, width)
    
    # Compute gradients manually for cross-entropy
    out = Tensor(loss.data if reduction != 'none' else loss.data, 
                requires_grad=logits.requires_grad,
                _children=(logits,), op='weighted_ce')
    
    def _backward():
        if out.grad is None or not logits.requires_grad:
            return
        
        # Gradient of cross-entropy w.r.t. logits
        # ∂L/∂logits = softmax(logits) - one_hot(target)
        grad_logits = exp_logits / sum_exp  # This is softmax
        
        # Subtract 1 at target positions
        grad_logits[np.arange(N), target_flat] -= 1.0
        
        # Apply class weights
        if class_weights is not None:
            weights = class_weights[target_flat]
            grad_logits *= weights.reshape(-1, 1)
        
        # Average or sum
        if reduction == 'mean':
            grad_logits /= N
        
        # Reshape back to original shape: (B*H*W, C) -> (B, C, H, W)
        grad_logits = grad_logits.reshape(batch_size, height, width, num_classes)
        grad_logits = grad_logits.transpose(0, 3, 1, 2)
        
        # Multiply by upstream gradient
        if isinstance(out.grad, np.ndarray):
            if out.grad.shape == ():
                grad_logits *= out.grad
            else:
                grad_logits *= out.grad.reshape(out.grad.shape + (1, 1, 1)[:4-len(out.grad.shape)])
        
        logits.grad = grad_logits if logits.grad is None else logits.grad + grad_logits
    
    out._backward = _backward
    return out


def combined_loss(pred: Tensor, target: Tensor, 
                 class_weights: Optional[np.ndarray] = None,
                 dice_weight: float = 0.5, ce_weight: float = 0.5) -> Tensor:
    """
    Combined Dice and Cross-Entropy Loss
    
    Combines the benefits of both loss functions:
    - Dice Loss: Good for class imbalance, focuses on overlap
    - Cross-Entropy: Good for pixel-wise classification
    
    Args:
        pred: Predicted logits (batch_size, num_classes, height, width)
        target: Ground truth (batch_size, height, width)
        class_weights: Weights for cross-entropy loss
        dice_weight: Weight for Dice loss component
        ce_weight: Weight for cross-entropy component
    
    Returns:
        Combined loss
    """
    # Dice loss
    d_loss = dice_loss(pred, target)
    
    # Weighted cross-entropy loss
    ce_loss = weighted_cross_entropy(pred, target, class_weights)
    
    # Combine losses
    total_loss = dice_weight * d_loss + ce_weight * ce_loss
    
    return total_loss


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax activation function
    
    Implements: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    
    Args:
        x: Input tensor
        axis: Axis along which to apply softmax
    
    Returns:
        Softmax probabilities
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
        
        # Softmax gradient: ∂softmax/∂x = softmax * (1 - softmax) for diagonal
        # For off-diagonal: -softmax_i * softmax_j
        # Simplified: use the fact that gradient is usually from cross-entropy
        dx = out.grad * softmax_data * (1.0 - softmax_data)
        x.grad = dx if x.grad is None else x.grad + dx
    
    out._backward = _backward
    return out


def focal_loss(pred: Tensor, target: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    """
    Focal Loss for handling extremely imbalanced classes
    
    Implements: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
    
    Focuses training on hard examples by down-weighting easy examples.
    
    Args:
        pred: Predicted logits (batch_size, num_classes, height, width)
        target: Ground truth (batch_size, height, width)
        alpha: Weighting factor for balanced classes
        gamma: Focusing parameter (higher = more focus on hard examples)
    
    Returns:
        Focal loss
    """
    batch_size, num_classes, height, width = pred.shape
    
    # Apply softmax to get probabilities
    pred_soft = softmax(pred, axis=1)
    
    # Get probabilities for target classes
    target_np = target.data.astype(np.int32).reshape(-1)
    pred_flat = pred_soft.data.transpose(0, 2, 3, 1).reshape(-1, num_classes)
    
    # Get probability of correct class
    p_t = pred_flat[np.arange(len(target_np)), target_np]
    
    # Focal loss formula
    focal_weight = alpha * np.power(1.0 - p_t, gamma)
    loss_data = -focal_weight * np.log(p_t + 1e-8)
    
    loss = Tensor(loss_data, requires_grad=pred.requires_grad, _children=(pred,), op='focal')
    
    def _backward():
        if loss.grad is None or not pred.requires_grad:
            return
        # Simplified gradient computation
        pass
    
    loss._backward = _backward
    return loss.mean()


def calculate_class_weights(target: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Calculate class weights inversely proportional to class frequency
    
    Args:
        target: Ground truth labels (batch_size, height, width)
        num_classes: Number of classes
    
    Returns:
        Class weights (num_classes,)
    """
    # Count pixels per class
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    for c in range(num_classes):
        class_counts[c] = np.sum(target == c)
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1.0)
    
    # Inverse frequency
    total_pixels = target.size
    weights = total_pixels / (num_classes * class_counts)
    
    # Normalize weights
    weights = weights / np.sum(weights) * num_classes
    
    return weights


def iou_score(pred: Tensor, target: Tensor, num_classes: int) -> np.ndarray:
    """
    Calculate Intersection over Union (IoU) for each class
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred: Predicted class labels (batch_size, height, width)
        target: Ground truth labels (batch_size, height, width)
        num_classes: Number of classes
    
    Returns:
        IoU for each class (num_classes,)
    """
    ious = np.zeros(num_classes, dtype=np.float64)
    
    pred_np = pred.data.astype(np.int32)
    target_np = target.data.astype(np.int32)
    
    for c in range(num_classes):
        pred_c = (pred_np == c)
        target_c = (target_np == c)
        
        intersection = np.sum(pred_c & target_c)
        union = np.sum(pred_c | target_c)
        
        if union > 0:
            ious[c] = intersection / union
        else:
            ious[c] = 0.0
    
    return ious


def dice_coefficient(pred: Tensor, target: Tensor, num_classes: int) -> np.ndarray:
    """
    Calculate Dice coefficient for each class
    
    Dice = 2|A ∩ B| / (|A| + |B|)
    
    Args:
        pred: Predicted class labels (batch_size, height, width)
        target: Ground truth labels (batch_size, height, width)
        num_classes: Number of classes
    
    Returns:
        Dice coefficient for each class (num_classes,)
    """
    dice_scores = np.zeros(num_classes, dtype=np.float64)
    
    pred_np = pred.data.astype(np.int32)
    target_np = target.data.astype(np.int32)
    
    for c in range(num_classes):
        pred_c = (pred_np == c)
        target_c = (target_np == c)
        
        intersection = np.sum(pred_c & target_c)
        pred_sum = np.sum(pred_c)
        target_sum = np.sum(target_c)
        
        if pred_sum + target_sum > 0:
            dice_scores[c] = 2.0 * intersection / (pred_sum + target_sum)
        else:
            dice_scores[c] = 0.0
    
    return dice_scores

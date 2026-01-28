from __future__ import annotations
import numpy as np
from .autograd import Tensor, log_softmax


def mse_loss(pred: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    diff = pred - target
    sq = diff * diff
    if reduction == 'mean':
        return sq.mean()
    elif reduction == 'sum':
        return sq.sum()
    else:
        return sq


def cross_entropy(logits: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Cross entropy with integer class targets.
    - logits: (N, C)
    - target: (N,) integer class ids, not requiring grad
    """
    if target.requires_grad:
        raise ValueError('Target for cross_entropy should not require grad')
    N, C = logits.data.shape
    # one-hot encode targets
    t_np = target.data.astype(np.int64)
    one_hot = np.zeros((N, C), dtype=np.float64)
    one_hot[np.arange(N), t_np] = 1.0
    oh = Tensor(one_hot, requires_grad=False)
    logp = log_softmax(logits, axis=1)
    nll = -(oh * logp).sum(axis=1)
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll

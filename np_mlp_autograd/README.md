# numpy MLP with autodiff (PyTorch-like)

A tiny dynamic autograd engine and MLP built with NumPy only (no PyTorch/TensorFlow/JAX). Supports:

- Tensor with automatic differentiation (reverse-mode), operator overloading
- Modules and Parameters (Linear, MLP), ReLU/Tanh/Sigmoid
- Losses: MSELoss, CrossEntropyLoss (stable log-softmax)
- Optimizer: SGD
- Training and inference demo on synthetic blobs

## Install

Requires Python 3.9+ and NumPy.

```bash
pip install -U numpy
```

No package install is needed; run from the repo root.

## Run the demo

```bash
python -m np_mlp_autograd.examples.train_demo
```

You'll see the loss decreasing and accuracy improving.

## API sketch

- `Tensor(data, requires_grad=False)`: wrap numpy array
- Ops: `+ - * / ** @`, `sum`, `mean`, `reshape`, `transpose`, `relu`, `tanh`, `sigmoid`, `exp`, `log`, `softmax`, `log_softmax`
- `Tensor.backward(grad=None)`: run reverse-mode autodiff
- `Module`, `Parameter`, `Linear`, `MLP`
- Losses: `mse_loss(pred, target)`, `cross_entropy(logits, target)`
- Optimizer: `SGD(params, lr)`

## Notes

- Broadcasting in gradients is supported for elementwise ops.
- Cross-entropy expects integer class labels (shape `(N,)`).
- This is educational code; it's not optimized for speed or memory.

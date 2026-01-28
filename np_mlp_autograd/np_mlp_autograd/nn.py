from __future__ import annotations
import math
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import numpy as np
from .autograd import Tensor, randn, tensor, relu


class Parameter(Tensor):
    def __init__(self, data, requires_grad: bool = True):
        super().__init__(np.array(data, dtype=np.float64), requires_grad=requires_grad)


class Module:
    def __init__(self):
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, Module] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault('_parameters', {})
            self._parameters[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault('_modules', {})
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # to be overridden
        raise NotImplementedError


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        # Kaiming uniform
        limit = math.sqrt(6 / in_features)
        W = (np.random.rand(in_features, out_features) * 2 - 1) * limit
        self.weight = Parameter(W)
        if bias:
            b = np.zeros((out_features,), dtype=np.float64)
            self.bias = Parameter(b)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class MLP(Module):
    def __init__(self, in_features: int, hidden_sizes: List[int], out_features: int, activation: str = 'relu'):
        super().__init__()
        sizes = [in_features] + hidden_sizes + [out_features]
        self.layers: List[Linear] = []
        for i in range(len(sizes) - 1):
            lin = Linear(sizes[i], sizes[i+1])
            self.layers.append(lin)
            setattr(self, f'lin_{i}', lin)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for i, lin in enumerate(self.layers):
            h = lin(h)
            if i < len(self.layers) - 1:
                if self.activation == 'relu':
                    h = h.relu()
                elif self.activation == 'tanh':
                    h = h.tanh()
                elif self.activation == 'sigmoid':
                    h = h.sigmoid()
                else:
                    raise ValueError(f'Unknown activation: {self.activation}')
        return h

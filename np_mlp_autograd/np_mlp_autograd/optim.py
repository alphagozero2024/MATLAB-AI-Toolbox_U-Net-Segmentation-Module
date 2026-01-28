from __future__ import annotations
from typing import Iterable, List
from .nn import Parameter


class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float = 0.1):
        self.params: List[Parameter] = list(params)
        self.lr = float(lr)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad

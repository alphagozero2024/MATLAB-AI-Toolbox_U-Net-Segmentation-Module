import numpy as np
from typing import Any, Callable, Iterable, Optional, Sequence, Set, Tuple, Union

Arrayable = Union[float, int, np.ndarray, 'Tensor']


def _to_ndarray(x: Arrayable) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.data
    if isinstance(x, (float, int)):
        return np.array(x, dtype=np.float64)
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    raise TypeError(f"Unsupported type: {type(x)}")


def tensor(x: Arrayable, requires_grad: bool = False) -> 'Tensor':
    return Tensor(_to_ndarray(x), requires_grad=requires_grad)


def zeros(shape: Union[int, Tuple[int, ...]], requires_grad: bool = False) -> 'Tensor':
    return Tensor(np.zeros(shape, dtype=np.float64), requires_grad=requires_grad)


def ones(shape: Union[int, Tuple[int, ...]], requires_grad: bool = False) -> 'Tensor':
    return Tensor(np.ones(shape, dtype=np.float64), requires_grad=requires_grad)


def randn(shape: Union[int, Tuple[int, ...]], requires_grad: bool = False, scale: float = 1.0) -> 'Tensor':
    return Tensor(np.random.randn(*((shape,) if isinstance(shape, int) else shape)).astype(np.float64) * scale, requires_grad=requires_grad)


class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False, _children: Tuple['Tensor', ...] = (), op: str = ""):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        if data.dtype != np.float64:
            data = data.astype(np.float64)
        self.data: np.ndarray = data
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(_children)
        self.op = op  # debug

    # ----- utility -----
    @property
    def shape(self):
        return self.data.shape

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data.item())

    def zero_grad(self):
        self.grad = None

    # ----- topological backward -----
    def backward(self, grad: Optional['Tensor'] = None):
        if not self.requires_grad:
            return
        # seed gradient
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensors")
            grad_out = np.ones_like(self.data)
        else:
            grad_out = _to_ndarray(grad)

        # build topo order
        topo = []
        visited = set()

        def build(v: 'Tensor'):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        # set final grad
        self.grad = (self.grad + grad_out) if self.grad is not None else grad_out
        # traverse
        for v in reversed(topo):
            v._backward()

    # ----- representation -----
    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, grad={(None if self.grad is None else np.array2string(self.grad))}, requires_grad={self.requires_grad})"

    # ----- helpers -----
    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        # sum gradients over broadcasted axes to match 'shape'
        if grad.shape == shape:
            return grad
        # If extra leading dims in grad, sum them
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0, keepdims=False)
        # For dims where original had size 1, sum along that axis
        for i, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
            if sdim == 1 and gdim != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    # ----- core ops -----
    def __add__(self, other: Arrayable) -> 'Tensor':
        other = other if isinstance(other, Tensor) else tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), op='+')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = Tensor._unbroadcast(out.grad, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = Tensor._unbroadcast(out.grad, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = _backward
        return out

    def __radd__(self, other: Arrayable) -> 'Tensor':
        return self.__add__(other)

    def __neg__(self) -> 'Tensor':
        out = Tensor(-self.data, requires_grad=self.requires_grad, _children=(self,), op='neg')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self.grad = (-out.grad) if self.grad is None else self.grad - out.grad
        out._backward = _backward
        return out

    def __sub__(self, other: Arrayable) -> 'Tensor':
        return self + (- (other if isinstance(other, Tensor) else tensor(other)))

    def __rsub__(self, other: Arrayable) -> 'Tensor':
        return (other if isinstance(other, Tensor) else tensor(other)) + (-self)

    def __mul__(self, other: Arrayable) -> 'Tensor':
        other = other if isinstance(other, Tensor) else tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), op='*')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad * other.data
                g = Tensor._unbroadcast(g, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = out.grad * self.data
                g = Tensor._unbroadcast(g, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = _backward
        return out

    def __rmul__(self, other: Arrayable) -> 'Tensor':
        return self.__mul__(other)

    def __truediv__(self, other: Arrayable) -> 'Tensor':
        other = other if isinstance(other, Tensor) else tensor(other)
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), op='/')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad / other.data
                g = Tensor._unbroadcast(g, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = -out.grad * self.data / (other.data ** 2)
                g = Tensor._unbroadcast(g, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = _backward
        return out

    def __rtruediv__(self, other: Arrayable) -> 'Tensor':
        return (other if isinstance(other, Tensor) else tensor(other)) / self

    def __pow__(self, power: float) -> 'Tensor':
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, _children=(self,), op=f'**{power}')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad * (power * (self.data ** (power - 1)))
            self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def __matmul__(self, other: Arrayable) -> 'Tensor':
        other = other if isinstance(other, Tensor) else tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad, _children=(self, other), op='@')

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                g = out.grad @ other.data.T
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = self.data.T @ out.grad
                other.grad = g if other.grad is None else other.grad + g
        out._backward = _backward
        return out

    # ----- reductions and shapes -----
    def sum(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, _children=(self,), op='sum')
        axis_tuple: Optional[Tuple[int, ...]]
        if axis is None:
            axis_tuple = None
        elif isinstance(axis, tuple):
            axis_tuple = axis
        else:
            axis_tuple = (axis,)

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad
            if axis_tuple is None:
                g = np.ones_like(self.data) * out.grad
            else:
                # expand grad to original shape
                shape = list(self.data.shape)
                if not keepdims:
                    for ax in axis_tuple:
                        g = np.expand_dims(g, ax)
                g = np.broadcast_to(g, self.data.shape)
            self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def mean(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> 'Tensor':
        if axis is None:
            denom = self.data.size
        else:
            if isinstance(axis, tuple):
                denom = 1
                for ax in axis:
                    denom *= self.data.shape[ax]
            else:
                denom = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / denom

    def reshape(self, *shape: int) -> 'Tensor':
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), op='reshape')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self.grad = out.grad.reshape(self.data.shape) if self.grad is None else self.grad + out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def T(self) -> 'Tensor':  # simple 2D transpose
        out = Tensor(self.data.T, requires_grad=self.requires_grad, _children=(self,), op='transpose')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            self.grad = out.grad.T if self.grad is None else self.grad + out.grad.T
        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> 'Tensor':
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad, _children=(self,), op='transpose')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            inv_axes = np.argsort(axes)
            self.grad = out.grad.transpose(*inv_axes) if self.grad is None else self.grad + out.grad.transpose(*inv_axes)
        out._backward = _backward
        return out

    # ----- nonlinearities -----
    def relu(self) -> 'Tensor':
        out_data = np.maximum(self.data, 0.0)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), op='relu')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad * (self.data > 0).astype(np.float64)
            self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def tanh(self) -> 'Tensor':
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), op='tanh')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad * (1 - t**2)
            self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def sigmoid(self) -> 'Tensor':
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), op='sigmoid')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad * s * (1 - s)
            self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def exp(self) -> 'Tensor':
        e = np.exp(self.data)
        out = Tensor(e, requires_grad=self.requires_grad, _children=(self,), op='exp')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad * e
            self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out

    def log(self) -> 'Tensor':
        l = np.log(self.data + 1e-12)
        out = Tensor(l, requires_grad=self.requires_grad, _children=(self,), op='log')

        def _backward():
            if out.grad is None or not self.requires_grad:
                return
            g = out.grad / (self.data + 1e-12)
            self.grad = g if self.grad is None else self.grad + g
        out._backward = _backward
        return out


# functional wrappers for convenience
relu = lambda x: x.relu()
tanh = lambda x: x.tanh()
sigmoid = lambda x: x.sigmoid()
softmax = lambda x, axis=-1: (x - x.data.max(axis=axis, keepdims=True)).exp() / (x - x.data.max(axis=axis, keepdims=True)).exp().sum(axis=axis, keepdims=True)

def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    # stable log-softmax
    m = x.data.max(axis=axis, keepdims=True)
    z = (x - m)
    logsumexp = z.exp().sum(axis=axis, keepdims=True).log()
    return z - logsumexp

"""Mesin diferensiasi otomatis mode mundur (reverse-mode autograd).

Setiap ``Tensor`` membungkus array NumPy dan, 

selama *forward pass*, mencatat
aturan gradien lokal yang dibutuhkan untuk *backward pass* melalui closure
``_backward``. Memanggil ``loss.backward()`` melakukan traversal topologis
pada graf komputasi dan mengakumulasikan ∂loss/∂param ke atribut ``.grad``
setiap tensor daun (parameter).

Perbedaan dengan implementasi non-autograd (``network.py``, ``layers.py``):
- Non-autograd: setiap lapisan menyimpan cache input secara manual
  (``self._input``, ``self._z``, ``self._a``) dan setiap fungsi aktivasi
  mengimplementasikan metode ``.derivative()`` yang ditulis tangan; gradien
  dihitung lapisan per lapisan di ``Dense.backward()``.
- Autograd (file ini): tidak ada ``.derivative()`` yang dipanggil secara
  eksplisit — gradien mengalir otomatis melalui graf komputasi yang dibangun
  saat forward pass.
"""

from __future__ import annotations

from typing import Callable
import numpy as np


class Tensor:
    """A NumPy-backed array node in a dynamic computation graph.

    Parameters
    ----------
    data : array-like
        The forward-pass values.
    _children : tuple[Tensor, ...]
        Parent tensors whose operation produced this tensor.
    _op : str
        Name of the operation that produced this tensor (for debugging).
    """

    # Avoids a per-instance __dict__, saving memory across large computation graphs.
    # fixed set of named attribute slots
    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(
        self,
        data,
        _children: tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev: tuple["Tensor", ...] = _children
        self._op: str = _op

    def zero_grad(self) -> None:
        """Reset accumulated gradient to zero."""
        self.grad = np.zeros_like(self.data)

    # ── Arithmetic operations ──────────────────────────────────────────

    def __add__(self, other: "Tensor | float | np.ndarray") -> "Tensor":
        other = _t(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _back() -> None:
            self.grad += _sum_to(out.grad, self.data.shape)
            other.grad += _sum_to(out.grad, other.data.shape)

        out._backward = _back
        return out

    def __radd__(self, other) -> "Tensor":
        # other + self  (scalar on the left)
        return self.__add__(other)

    def __sub__(self, other: "Tensor | float | np.ndarray") -> "Tensor":
        other = _t(other)
        out = Tensor(self.data - other.data, (self, other), "-")

        def _back() -> None:
            self.grad += _sum_to(out.grad, self.data.shape)
            other.grad -= _sum_to(out.grad, other.data.shape)

        out._backward = _back
        return out

    def __rsub__(self, other) -> "Tensor":
        # other - self  (scalar on the left, e.g. 1.0 - tensor)
        return _t(other).__sub__(self)

    def __neg__(self) -> "Tensor":
        # unary minus: -self
        out = Tensor(-self.data, (self,), "neg")

        def _back() -> None:
            self.grad -= out.grad

        out._backward = _back
        return out

    def __mul__(self, other: "Tensor | float | np.ndarray") -> "Tensor":
        other = _t(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _back() -> None:
            self.grad += _sum_to(other.data * out.grad, self.data.shape)
            other.grad += _sum_to(self.data * out.grad, other.data.shape)

        out._backward = _back
        return out

    def __rmul__(self, other) -> "Tensor":
        # other * self  (scalar on the left)
        return self.__mul__(other)

    def __truediv__(self, other: "Tensor | float | np.ndarray") -> "Tensor":
        other = _t(other)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _back() -> None:
            self.grad += _sum_to(out.grad / other.data, self.data.shape)
            other.grad += _sum_to(
                -self.data * out.grad / (other.data ** 2), other.data.shape
            )

        out._backward = _back
        return out

    def __rtruediv__(self, other) -> "Tensor":
        # other / self  (scalar on the left)
        return _t(other).__truediv__(self)

    def __matmul__(self, other: "Tensor | np.ndarray") -> "Tensor":
        """Matrix multiplication  A @ B.

        Gradient rules:
          dL/dA = grad_out @ B.T
          dL/dB = A.T @ grad_out
        """
        other = _t(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _back() -> None:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _back
        return out

    def __pow__(self, exp: int | float) -> "Tensor":
        # self ** exp  (scalar exponent only)
        assert isinstance(exp, (int, float)), "Only scalar exponents are supported."
        out = Tensor(self.data ** exp, (self,), f"**{exp}")

        def _back() -> None:
            self.grad += exp * self.data ** (exp - 1) * out.grad

        out._backward = _back
        return out

    # ── Reduction operations ───────────────────────────────────────────

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), (self,), "sum")

        def _back() -> None:
            g = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                for ax in sorted(axes):
                    g = np.expand_dims(g, axis=ax)
            self.grad += np.broadcast_to(g, self.data.shape).copy()

        out._backward = _back
        return out

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        """Average over the specified axis (or all elements if axis=None)."""
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ── Elementwise math ──────────────────────────────────────────────

    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _back() -> None:
            # d/dx exp(x) = exp(x)
            self.grad += e * out.grad

        out._backward = _back
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log")

        def _back() -> None:
            # d/dx ln(x) = 1/x
            self.grad += out.grad / self.data

        out._backward = _back
        return out

    def clip(self, lo: float, hi: float) -> "Tensor":
        """Clip values; gradient is zero where data falls outside [lo, hi]."""
        out = Tensor(np.clip(self.data, lo, hi), (self,), "clip")

        def _back() -> None:
            mask = (self.data >= lo) & (self.data <= hi)
            self.grad += out.grad * mask.astype(np.float64)

        out._backward = _back
        return out

    # ── Activation operations ──────────────────────────────────────────
    #
    # Each method records the forward value and closes over it in _back.
    # No external .derivative() is ever called — this IS the autograd.

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _back() -> None:
            # d/dx max(0, x) = 1 if x > 0, else 0
            self.grad += (self.data > 0.0).astype(np.float64) * out.grad

        out._backward = _back
        return out

    def leaky_relu(self, alpha: float = 0.01) -> "Tensor":
        out = Tensor(
            np.where(self.data > 0, self.data, alpha * self.data), (self,), "leaky_relu"
        )

        def _back() -> None:
            self.grad += np.where(self.data > 0, 1.0, alpha) * out.grad

        out._backward = _back
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _back() -> None:
            # σ'(x) = σ(x) · (1 − σ(x))
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _back
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _back() -> None:
            # tanh'(x) = 1 − tanh²(x)
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _back
        return out

    def softmax(self) -> "Tensor":
        """Numerically-stable softmax along the last axis.

        Full-Jacobian backward:
          dL/dz_i = s_i · (dL/ds_i − Σ_j dL/ds_j · s_j)
        """
        shifted = self.data - np.max(self.data, axis=-1, keepdims=True)
        e = np.exp(shifted)
        s = e / np.sum(e, axis=-1, keepdims=True)
        out = Tensor(s, (self,), "softmax")

        def _back() -> None:
            dot = np.sum(out.grad * s, axis=-1, keepdims=True)
            self.grad += s * (out.grad - dot)

        out._backward = _back
        return out

    def swish(self, beta: float = 1.0) -> "Tensor":
        """Swish(x) = x · σ(β·x)."""
        sig = 1.0 / (1.0 + np.exp(-beta * self.data))
        out = Tensor(self.data * sig, (self,), "swish")

        def _back() -> None:
            # swish'(x) = σ(β·x) + β·x·σ(β·x)·(1 − σ(β·x))
            d = sig + beta * self.data * sig * (1.0 - sig)
            self.grad += d * out.grad

        out._backward = _back
        return out

    # ── Graph traversal ───────────────────────────────────────────────

    def backward(self) -> None:
        """Back-propagate gradients through the entire computation graph.

        1. Build a reverse topological ordering of all reachable nodes.
        2. Seed this (loss) tensor's gradient with ones.
        3. Call each node's ``_backward`` closure in reverse order —
           each closure accumulates the chain-rule contribution into its
           parent tensors' ``.grad`` arrays.
        """
        topo: list[Tensor] = []
        visited: set[int] = set()

        def _build(v: "Tensor") -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    _build(child)
                topo.append(v)

        _build(self)

        # Seed: dL/dL = 1
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, op='{self._op}')"


# ── Private helpers ────────────────────────────────────────────────────────


def _t(x) -> Tensor:
    """Wrap *x* in a Tensor if it is not already one."""
    return x if isinstance(x, Tensor) else Tensor(x)


def _sum_to(g: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum gradient *g* over broadcast-introduced dimensions to match *shape*."""
    if g.shape == shape:
        return g
    # Remove extra leading dimensions (broadcasting added them)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    # Sum over axes that were size-1 in the original
    for i, s in enumerate(shape):
        if s == 1:
            g = g.sum(axis=i, keepdims=True)
    return g.copy()

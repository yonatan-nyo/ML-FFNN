"""Autograd variant of the Dense layer."""

from __future__ import annotations
import numpy as np

from .activations import Activation, get_activation
from .initializers import Initializer, XavierInitializer, get_initializer
from .autograd import Tensor


class AutogradDense:
    """Fully-connected layer whose weights are ``Tensor`` objects.

    ``forward`` extends the computation graph; call ``loss.backward()`` on the
    network output and ``W.grad`` / ``b.grad`` are populated automatically.
    No explicit ``backward`` method required.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Activation | str = "linear",
        initializer: Initializer | str | None = None,
        seed: int | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        if isinstance(activation, str):
            activation = get_activation(activation)
        self.activation: Activation = activation

        if initializer is None:
            initializer = XavierInitializer(seed=seed)
        elif isinstance(initializer, str):
            initializer = get_initializer(initializer, seed=seed)

        self.W = Tensor(initializer((input_dim, output_dim)))
        self.b = Tensor(np.zeros((1, output_dim)))

    def forward(self, X: Tensor) -> Tensor:
        z = X @ self.W + self.b
        return self.activation.forward_tensor(z)

    def zero_grad(self) -> None:
        self.W.zero_grad()
        self.b.zero_grad()

    def num_params(self) -> int:
        return self.W.data.size + self.b.data.size

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation.name(),
            "weights": self.W.data.tolist(),
            "bias": self.b.data.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AutogradDense":
        layer = cls(d["input_dim"], d["output_dim"], activation=d["activation"])
        layer.W = Tensor(np.array(d["weights"]))
        layer.b = Tensor(np.array(d["bias"]))
        return layer

    def __repr__(self) -> str:
        return (
            f"AutogradDense(in={self.input_dim}, out={self.output_dim}, "
            f"act={self.activation.name()}, params={self.num_params()})"
        )

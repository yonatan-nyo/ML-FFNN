"""Dense (fully-connected) layer."""

from __future__ import annotations
import numpy as np

from .activations import Activation, Linear, get_activation
from .initializers import Initializer, XavierInitializer, get_initializer


class Dense:
    """A single fully-connected layer: output = activation(X @ W + b)

    Parameters
    ----------
    input_dim  : number of input features (fan_in)
    output_dim : number of neurons in this layer (fan_out)
    activation : Activation instance **or** name string
    initializer: Initializer instance **or** name string (for weights)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Activation | str = "linear",
        initializer: Initializer | str | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Resolve activation
        if isinstance(activation, str):
            activation = get_activation(activation)
        self.activation: Activation = activation

        # Resolve initializer (default: Xavier)
        if initializer is None:
            initializer = XavierInitializer(seed=None)
        elif isinstance(initializer, str):
            initializer = get_initializer(initializer)
        self.initializer: Initializer = initializer

        # Weights: shape (input_dim, output_dim)
        self.weights: np.ndarray = self.initializer((input_dim, output_dim))
        # Bias: shape (1, output_dim)
        self.bias: np.ndarray = np.zeros((1, output_dim))

        # Gradients (same shape as weights/bias)
        self.grad_weights: np.ndarray = np.zeros_like(self.weights)
        self.grad_bias: np.ndarray = np.zeros_like(self.bias)

        # Cache for forward/backward
        self._input: np.ndarray | None = None   # input to this layer
        self._z: np.ndarray | None = None       # pre-activation: X @ W + b
        self._a: np.ndarray | None = None       # post-activation: activation(z)

    # ─── forward ──────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        """X shape: (batch, input_dim) -> returns (batch, output_dim)"""
        self._input = X
        self._z = X @ self.weights + self.bias
        self._a = self.activation.forward(self._z)
        return self._a

    # ─── backward ─────────────────────────────────────
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Given dL/d(activation_output), compute and store gradients,
        then return dL/d(input) for the previous layer.

        d_out shape: (batch, output_dim)
        returns: (batch, input_dim)
        """
        # dL/dz = dL/da * da/dz (element-wise for non-softmax)
        d_z = d_out * self.activation.derivative(self._z)  # (batch, output_dim)

        batch_size = d_out.shape[0]

        # dL/dW = X^T @ d_z  (averaged over batch)
        self.grad_weights = self._input.T @ d_z / batch_size
        # dL/db = mean of d_z over batch
        self.grad_bias = np.mean(d_z, axis=0, keepdims=True)
        # dL/d(input) = d_z @ W^T
        d_input = d_z @ self.weights.T
        return d_input

    # ─── helpers ──────────────────────────────────────
    def num_params(self) -> int:
        return self.weights.size + self.bias.size

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation.name(),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Dense":
        layer = cls(d["input_dim"], d["output_dim"], activation=d["activation"])
        layer.weights = np.array(d["weights"])
        layer.bias = np.array(d["bias"])
        return layer

    def __repr__(self) -> str:
        return (
            f"Dense(in={self.input_dim}, out={self.output_dim}, "
            f"act={self.activation.name()}, params={self.num_params()})"
        )

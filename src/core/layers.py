"""Neural network layers (Dense, RMSNorm, etc.)."""

from __future__ import annotations
import numpy as np

from .activations import Activation, get_activation
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
        seed: int | None = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Resolve activation
        if isinstance(activation, str):
            activation = get_activation(activation)
        self.activation: Activation = activation

        # Resolve initializer (default: Xavier)
        if initializer is None:
            initializer = XavierInitializer(seed=seed)
        elif isinstance(initializer, str):
            initializer = get_initializer(initializer, seed=seed)
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

        # d_out already carries the 1/N factor from loss.backward(), so we sum
        # (not mean) here to match the autograd engine which also accumulates a sum.
        # dL/dW = X^T @ d_z
        self.grad_weights = self._input.T @ d_z
        # dL/db = sum of d_z over batch (d_out already has the 1/N from the loss)
        self.grad_bias = np.sum(d_z, axis=0, keepdims=True)
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


class RMSNorm:
    """Root Mean Square Normalization (RMSNorm) Layer.
    https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html
    
    Parameters
    ----------
    input_dim : int
        Number of features to normalize.
    eps : float
        A small constant for numerical stability. Default is 1e-8.
    """

    def __init__(self, input_dim: int, eps: float = 1e-8):
        self.input_dim = input_dim
        self.eps = eps
        
        # Learnable scale parameter (gamma)
        self.gamma: np.ndarray = np.ones((1, input_dim))
        self.grad_gamma: np.ndarray = np.zeros_like(self.gamma)
        
        # We don't use bias in standard RMSNorm as per the paper, only scale.
        
        # Caches for backward pass
        self._input: np.ndarray | None = None
        self._rms: np.ndarray | None = None
        self._normalized: np.ndarray | None = None
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for RMSNorm. X shape: (batch, input_dim)"""
        self._input = X
        
        # Calculate RMS for each sample in the batch
        # RMS = sqrt( 1/d * sum(X^2) + eps )
        self._rms = np.sqrt(np.mean(X**2, axis=-1, keepdims=True) + self.eps)
        
        # Normalize
        self._normalized = X / self._rms
        
        # Scale
        out = self._normalized * self.gamma
        return out
        
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """Backward pass for RMSNorm. 
        d_out shape: (batch, input_dim)
        """
        # Gradient w.r.t gamma
        self.grad_gamma = np.sum(d_out * self._normalized, axis=0, keepdims=True)
        
        # Gradient w.r.t normalized input
        d_norm = d_out * self.gamma
        
        # Gradient w.r.t input X (vectorized derivation)
        # d_norm_dX = 1/RMS - X / (RMS^3 * D) * X
        # dL/dX = dL/d_norm * d_norm/dX
        term1 = d_norm / self._rms
        term2 = (self._normalized / self._rms) * np.mean(d_norm * self._normalized, axis=-1, keepdims=True)
        
        d_input = term1 - term2
        return d_input

    def num_params(self) -> int:
        return self.gamma.size

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "eps": self.eps,
            "gamma": self.gamma.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RMSNorm":
        layer = cls(d["input_dim"], eps=d["eps"])
        layer.gamma = np.array(d["gamma"])
        return layer

    def __repr__(self) -> str:
        return f"RMSNorm(in={self.input_dim}, eps={self.eps}, params={self.num_params()})"

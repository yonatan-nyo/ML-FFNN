"""Activation functions"""

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np

class Activation(ABC):
    """Base class for activation functions."""

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Return the derivative evaluated at *z* (the pre-activation value)."""
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.name()


# ──────────────────────────────────────────────
# Concrete activations
# ──────────────────────────────────────────────


class Linear(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return z * 0.0 + 1.0  # ones with same shape

    def name(self) -> str:
        return "linear"


class ReLU(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, 0.0)

    def name(self) -> str:
        return "relu"


class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # σ'(z) = σ(z) · (1 − σ(z))
        s = self.forward(z)
        return s * (1.0 - s)

    def name(self) -> str:
        return "sigmoid"


class Tanh(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return 1.0 - np.tanh(z) ** 2

    def name(self) -> str:
        return "tanh"


class Softmax(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        # softmax with max-shift for numerical stability; supports batches
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        """Return the Jacobian diagonal (used only when softmax is NOT paired
        with categorical cross-entropy — the common paired case is handled
        directly in the loss backward pass).
        For the diagonal: dS_i/dz_i = S_i * (1 - S_i)
        """
        s = self.forward(z)
        return s * (1.0 - s)

    def name(self) -> str:
        return "softmax"


# ──────────────────────────────────────────────
# Bonus activations (2 extra — spec bonus 5%)
# ──────────────────────────────────────────────


class LeakyReLU(Activation):
    """https://www.geeksforgeeks.org/machine-learning/leaky-relu-activation-function-in-deep-learning/"""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, self.alpha * z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, self.alpha)

    def name(self) -> str:
        return f"leaky_relu(alpha={self.alpha})"


class Swish(Activation):
    """https://www.geeksforgeeks.org/deep-learning/swish-activation-function/"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, z: np.ndarray) -> np.ndarray:
        # swish(z) = x * sigmoid(βx)
        return z * self._sigmoid(self.beta * z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # swish'(z) = σ(β·z) + β·z·σ(β·z)·(1 − σ(β·z))
        sig = self._sigmoid(self.beta * z)
        return sig + self.beta * z * sig * (1.0 - sig)

    def name(self) -> str:
        return f"swish(beta={self.beta})"


# Helper to look up by name string
_activations: dict[str, type[Activation]] = {
    "linear": Linear,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
    "leaky_relu": LeakyReLU,
    "swish": Swish,
}


ActivationName = Literal["linear", "relu", "sigmoid", "tanh", "softmax", "leaky_relu", "swish"]


def get_activation(name: ActivationName) -> Activation:
    """Return an activation instance by name string."""
    return _activations[name]()

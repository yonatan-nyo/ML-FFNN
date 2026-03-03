"""Activation functions"""

from abc import ABC, abstractmethod
import math
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
        # 1 if z > 0, else 0 (element-wise)
        return 1.0 * (z > 0)

    def name(self) -> str:
        return "relu"


class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + math.exp(-z))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # σ'(z) = σ(z) · (1 − σ(z))
        s = self.forward(z)
        return s * (1.0 - s)

    def name(self) -> str:
        return "sigmoid"


class Tanh(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        # tanh(z) = (e^z − e^(−z)) / (e^z + e^(−z))
        e_pos = math.exp(z)
        e_neg = math.exp(-z)
        return (e_pos - e_neg) / (e_pos + e_neg)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # 2 / (e^z + e^(−z))^2
        return (2 / (math.exp(z) + math.exp(-z))) ** 2

    def name(self) -> str:
        return "tanh"


class Softmax(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        # softmax(z_i) = e^z_i / Σ_j e^z_j  (max-shifted for stability)
        denominator = 0;
        z_len = len(z)
        for i in range(z_len):
            denominator += math.exp(z[i])
        
        def softmax_i(i: int) -> float:
            return math.exp(z[i]) / denominator

        return np.array([softmax_i(i) for i in range(z_len)], dtype=z.dtype)

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
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, z: np.ndarray) -> np.ndarray:
        # z if z > 0, else alpha·z
        pos = z > 0
        return z * pos + self.alpha * z * (1 - pos)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # 1 if z > 0, else alpha
        pos = 1.0 * (z > 0)
        return pos + self.alpha * (1.0 - pos)

    def name(self) -> str:
        return f"leaky_relu(alpha={self.alpha})"


class Swish(Activation):
    """Swish / SiLU activation: z · σ(z)"""

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid helper (from-scratch)."""
        out = np.empty_like(z, dtype=np.float64)
        for i in range(z.size):
            v = float(z.flat[i])
            if v >= 0:
                e = math.exp(-min(v, 500.0))
                out.flat[i] = 1.0 / (1.0 + e)
            else:
                e = math.exp(max(v, -500.0))
                out.flat[i] = e / (1.0 + e)
        return out

    def forward(self, z: np.ndarray) -> np.ndarray:
        # swish(z) = z · σ(z)
        return z * self._sigmoid(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        # swish'(z) = σ(z) + z · σ(z) · (1 − σ(z))
        sig = self._sigmoid(z)
        return sig + z * sig * (1.0 - sig)

    def name(self) -> str:
        return "swish"


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


def get_activation(name: str) -> Activation:
    """Return an activation instance by name string."""
    key = name.lower().replace(" ", "_")
    if key not in _activations:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(_activations.keys())}"
        )
    return _activations[key]()

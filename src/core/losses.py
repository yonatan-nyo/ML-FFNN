"""Loss functions implemented with NumPy for batch operations."""

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """Base class for loss functions."""

    @abstractmethod
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the scalar loss averaged over the batch."""
        ...

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Return dL/d(y_pred) with the same shape as y_pred."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    def __repr__(self) -> str:
        return self.name()


class MSE(Loss):
    """Mean Squared Error: L = (1/N) * sum((y_true - y_pred)^2)"""

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.shape[0]
        return 2.0 * (y_pred - y_true) / n

    def name(self) -> str:
        return "mse"


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy: L = -(1/N) * sum(y*ln(p) + (1-y)*ln(1-p))"""

    _eps = 1e-12  # clipping for numerical stability

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, self._eps, 1.0 - self._eps)
        return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, self._eps, 1.0 - self._eps)
        n = y_true.shape[0]
        return (-(y_true / p) + (1.0 - y_true) / (1.0 - p)) / n

    def name(self) -> str:
        return "binary_cross_entropy"


class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy: L = -(1/N) * sum_i sum_c y_ic * ln(p_ic)"""

    _eps = 1e-12

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        p = np.clip(y_pred, self._eps, 1.0)
        return float(-np.mean(np.sum(y_true * np.log(p), axis=-1)))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, self._eps, 1.0)
        n = y_true.shape[0]
        return -(y_true / p) / n

    def name(self) -> str:
        return "categorical_cross_entropy"


# Helper to look up by name
_losses: dict[str, type[Loss]] = {
    "mse": MSE,
    "binary_cross_entropy": BinaryCrossEntropy,
    "categorical_cross_entropy": CategoricalCrossEntropy,
}


def get_loss(name: str) -> Loss:
    key = name.lower().replace(" ", "_")
    if key not in _losses:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_losses.keys())}")
    return _losses[key]()

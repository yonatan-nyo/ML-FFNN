from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

class BaseModel(ABC):
    """Abstract base class for all machine learning models."""

    @abstractmethod
    def fit(
        self,
        x_train: List[List[float]],
        y_train: List[List[float]],
        val_data: Optional[Tuple] = None,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 100,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Returns
        -------
        dict with keys 'train_loss' and 'val_loss' (list of floats per epoch).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: List[List[float]]) -> List[List[float]]:
        """Run inference on X and return predictions."""
        raise NotImplementedError

    @abstractmethod
    def plot_weight_distribution(self, layers: Optional[List[int]] = None) -> None:
        """Plot the weight distribution for the given layer indices."""
        raise NotImplementedError

    @abstractmethod
    def plot_gradient_distribution(self, layers: Optional[List[int]] = None) -> None:
        """Plot the gradient distribution for the given layer indices."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model parameters to a JSON file."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """Restore a model from a JSON file."""
        raise NotImplementedError


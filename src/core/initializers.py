"""Weight initialisation strategies."""

from abc import ABC, abstractmethod
from typing import Literal
import numpy as np


class Initializer(ABC):
    @abstractmethod
    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    def __repr__(self) -> str:
        return self.name()


class ZeroInitializer(Initializer):
    """All weights set to zero."""

    def __init__(self, seed: int | None = None):
        pass 

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape)

    def name(self) -> str:
        return "zero"


class UniformInitializer(Initializer):
    """Random uniform in [low, high)."""

    def __init__(self, low: float = -1.0, high: float = 1.0, seed: int | None = None):
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        return self.rng.uniform(self.low, self.high, size=shape)

    def name(self) -> str:
        return f"uniform(low={self.low}, high={self.high})"


class NormalInitializer(Initializer):
    """Random normal with given mean and variance."""

    def __init__(self, mean: float = 0.0, variance: float = 1.0, seed: int | None = None):
        self.mean = mean
        self.variance = variance
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        std = np.sqrt(self.variance)
        return self.rng.normal(self.mean, std, size=shape)

    def name(self) -> str:
        return f"normal(mean={self.mean}, var={self.variance})"


# ──────────────────────────────────────────────
# Bonus initializers (spec bonus 5%)
# ──────────────────────────────────────────────

class XavierInitializer(Initializer):
    """Xavier / Glorot initialisation: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))"""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=shape)

    def name(self) -> str:
        return "xavier"


class HeInitializer(Initializer):
    """He initialisation: N(0, 2/fan_in)"""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, shape: tuple[int, ...]) -> np.ndarray:
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return self.rng.normal(0, std, size=shape)

    def name(self) -> str:
        return "he"


# Helper to look up by name
_initializers: dict[str, type[Initializer]] = {
    "zero": ZeroInitializer,
    "uniform": UniformInitializer,
    "normal": NormalInitializer,
    "xavier": XavierInitializer,
    "he": HeInitializer,
}


InitializerName = Literal["zero", "uniform", "normal", "xavier", "he"]


def get_initializer(name: InitializerName, **kwargs) -> Initializer:
    return _initializers[name](**kwargs)

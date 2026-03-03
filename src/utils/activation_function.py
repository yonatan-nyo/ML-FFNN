import math
from typing import List


class ActivationFunction:
    @staticmethod
    def linear(x: float) -> float:
        return x

    @staticmethod
    def reLU(x: float) -> float:
        return max(0, x)

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def tanh(x: float) -> float:
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    @staticmethod
    def hyperbolic_tangent(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    def softmax(x: List[float]) -> List[float]:
        def softmax_i(x_i: float, denominator: float) -> float:
            return math.exp(x_i) / denominator

        denominator = sum(math.exp(x_i) for x_i in x)
        return [softmax_i(x_i, denominator) for x_i in x]

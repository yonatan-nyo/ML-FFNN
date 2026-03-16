import math
from typing import List


class ActivationFunction:
    @staticmethod
    def linear(x: float) -> float:
        return x

    @staticmethod
    def derivative_linear(x: float) -> float:
        return 1

    @staticmethod
    def reLU(x: float) -> float:
        return max(0, x)

    @staticmethod
    def derivative_reLU(x: float) -> float:
        return 1 if x > 0 else 0

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def derivative_sigmoid(x: float) -> float:
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x: float) -> float:
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

    @staticmethod
    def derivative_tanh(x: float) -> float:
        denominator = (math.exp(x) - math.exp(-x))**2
        return 4 / denominator

    @staticmethod
    def hyperbolic_tangent(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    def derivative_hyperbolic_tangent(x: float) -> float:
        return ActivationFunction.derivative_tanh(x)

    @staticmethod
    def softmax(x: List[float]) -> List[float]:
        def softmax_i(x_i: float, denominator: float) -> float:
            return math.exp(x_i) / denominator

        denominator = sum(math.exp(x_i) for x_i in x)
        return [softmax_i(x_i, denominator) for x_i in x]

    @staticmethod
    def derivative_softmax(x: List[float]) -> List[List[float]]:
        s = ActivationFunction.softmax(x)
        jacobian = []
        for i in range(len(s)):
            row = []
            for j in range(len(s)):
                if i == j:
                    row.append(s[i] * (1 - s[i]))
                else:
                    row.append(-s[i] * s[j])
            jacobian.append(row)
        return jacobian

import math
from typing import List

"""
Binary cross-entropy merupakan kasus khusus categorical cross-entropy dengan kelas sebanyak 2.
Log yang digunakan merupakan logaritma natural (logaritma dengan basis e).
"""


class LossFunction:
    @staticmethod
    def MSE(y_true: List[float], y_pred: List[float]) -> float:
        y_list_length = len(y_true)
        if (y_list_length != len(y_pred)):
            raise ValueError("Length of y_true and y_pred must be the same.")

        ret = 0
        for i in range(y_list_length):
            ret += ((y_true[i] - y_pred[i]) ** 2)/y_list_length
        return ret

    @staticmethod
    def derivative_MSE(y_true: List[float], y_pred: List[float]) -> List[float]:
        y_list_length = len(y_true)
        if (y_list_length != len(y_pred)):
            raise ValueError("Length of y_true and y_pred must be the same.")
        
        ret = []
        for i in range(y_list_length):
            # Derivative of MSE: 2 * (y_pred - y_true) / N
            grad = 2 * (y_pred[i] - y_true[i])
            ret.append(grad / y_list_length)
        return ret

    @staticmethod
    def binary_cross_entropy(y_true: List[float], y_pred: List[float]) -> float:
        y_list_length = len(y_true)
        if (y_list_length != len(y_pred)):
            raise ValueError("Length of y_true and y_pred must be the same.")

        ret = 0
        for i in range(y_list_length):
            ret += -(1/y_list_length)*(y_true[i] * math.log(y_pred[i]) +
                                       (1 - y_true[i]) * math.log(1 - y_pred[i]))
        return ret

    @staticmethod
    def derivative_binary_cross_entropy(y_true: List[float], y_pred: List[float
                                                                          ]) -> List[float]:
        y_list_length = len(y_true)
        if (y_list_length != len(y_pred)):
            raise ValueError("Length of y_true and y_pred must be the same.")

        ret = []
        eps = 1e-15 # Mencegah ZeroDivisionError
        for i in range(y_list_length):
            y_p = max(min(y_pred[i], 1 - eps), eps)
            y_t = y_true[i]
            
            # Turunan BCE: (-y/y_pred + (1-y)/(1-y_pred)) / N
            grad = -(y_t / y_p) + ((1 - y_t) / (1 - y_p))
            ret.append(grad / y_list_length)
        return ret

    @staticmethod
    def categorical_cross_entropy(y_true: List[List[float]], y_pred: List[List[float]]) -> float:
        y_list_length = len(y_true)
        if (y_list_length != len(y_pred)):
            raise ValueError("Length of y_true and y_pred must be the same.")

        ret = 0
        for i in range(y_list_length):
            category_count = len(y_true[i])
            if (len(y_true[i]) != len(y_pred[i])):
                raise ValueError(
                    "Number of categories in y_true and y_pred must be the same.")

            for j in range(category_count):
                ret += -(1/y_list_length)*y_true[i][j] * math.log(y_pred[i][j])
        return ret

    @staticmethod
    def derivative_categorical_cross_entropy(y_true: List[List[float]], y_pred: List[List[float]]) -> List[List[float]]:
        y_list_length = len(y_true)
        if (y_list_length != len(y_pred)):
            raise ValueError("Length of y_true and y_pred must be the same.")

        ret = []
        eps = 1e-15 # Mencegah ZeroDivisionError
        for i in range(y_list_length):
            category_count = len(y_true[i])
            if (len(y_true[i]) != len(y_pred[i])):
                raise ValueError("Number of categories in y_true and y_pred must be the same.")

            row_ret = []
            for j in range(category_count):
                y_p = max(min(y_pred[i][j], 1 - eps), eps)
                y_t = y_true[i][j]
                
                # Turunan CCE: (-y/y_pred) / N
                grad = -(y_t / y_p)
                row_ret.append(grad / y_list_length)
            ret.append(row_ret)
        return ret

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

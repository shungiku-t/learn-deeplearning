from dezero.core import Variable
from dezero.functions import sum_to


def mean_squared_error(y0: Variable, y1: Variable) -> Variable:
    """平均二乗誤差"""
    diff = y0 - y1
    return sum_to(diff ** 2) / len(y0)

from dezero.core import Variable, exp


def sigmoid(x: Variable) -> Variable:
    y = 1 / (1 + exp(-x))
    return y

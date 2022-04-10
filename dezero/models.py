from typing import Optional
from dezero.activations import sigmoid

from dezero.core import Variable
from dezero.functions import matmul
from dezero.utils import as_variable


def linear_simple(x: Variable, W: Variable, b: Optional[Variable] = None) -> Variable:
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None
    return y


def nuralnet_simple(
    x: Variable, W1: Variable, W2: Variable, b1: Variable, b2: Variable
) -> Variable:
    x = as_variable(x)
    y: Variable = linear_simple(x, W1, b1)
    y: Variable = sigmoid(y)
    y: Variable = linear_simple(y, W2, b2)
    return y

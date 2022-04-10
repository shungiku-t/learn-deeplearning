import numpy as np

from dezero.core import Variable


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(obj) -> Variable:
    if isinstance(obj, Variable):
        return obj
    elif isinstance(obj, np.ndarray):
        return Variable(obj)
    else:
        raise TypeError


def pprint(var):
    print("--------")
    print("type: ", type(var))
    print("val: ", var)
    print("--------")

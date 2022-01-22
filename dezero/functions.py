from __future__ import annotations
from typing import Tuple, Union
import numpy as np
from dezero.core_simple import Function, Variable


class Square(Function):
    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        y = x * x
        return y

    def backward(self, gy: np.ndarray):
        return 2 * self.inputs[0].data * gy


def square(x: Variable, name="") -> Variable:
    y = Square(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Reshape(Function):
    def __init__(self, shape: tuple, name=None) -> None:
        self.shape = shape
        self.x_shape = None  # 入力のshapeを記憶する
        self.name = name

    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy: np.ndarray) -> Union[Tuple[np.ndarray], np.ndarray]:
        if self.x_shape is None:
            raise ValueError
        return gy.reshape(self.x_shape)


def reshape(x: Variable, shape: tuple, name="") -> Variable:
    y = Reshape(shape, name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y

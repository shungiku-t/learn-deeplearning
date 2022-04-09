from __future__ import annotations
from typing import Tuple, Union
import numpy as np
from dezero.core import Function, Variable


class Square(Function):
    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        y = x * x
        return y

    # def backward(self, gy: np.ndarray):
    #     return 2 * self.inputs[0].data * gy

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        return 2 * self.inputs[0].data * gy


def square(x: Variable, name="") -> Variable:
    y = Square(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Sin(Function):
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        y = np.sin(x)
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        return cos(self.inputs[0]) * gy


def sin(x: Variable, name="") -> Variable:
    y = Sin(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Cos(Function):
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        y = np.cos(x)
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        return -sin(self.inputs[0]) * gy


def cos(x: Variable, name="") -> Variable:
    y = Cos(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Reshape(Function):
    def __init__(self, shape: tuple, name=None) -> None:
        super().__init__(name)
        self.shape = shape
        self.x_shape = None  # 入力のshapeを記憶する

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


class Transpose(Function):
    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        y = x.T
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        return transpose(gy)


def transpose(x: Variable, name="") -> Variable:
    y = Transpose(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


# class BroadCastTo(Function):
#     def __init__(self, shape: tuple, name=None) -> None:
#         super().__init__(name)
#         self.shape = shape
#         self.x_shape = None  # 順伝搬時に入力のshapeを記憶する

#     def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
#         if len(xs) != 1:
#             raise ValueError
#         x = xs[0]
#         y = x.T
#         return y

#     def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
#         # if len(gys) != 1:
#         #     raise ValueError
#         # gy = gys[0]
#         # return transpose(gy)
#         pass


# def broad_cast_to(x: Variable, shape: tuple, name="") -> Variable:
#     y = BroadCastTo(shape, name)
#     if not isinstance(y, Variable):
#         raise TypeError
#     return y

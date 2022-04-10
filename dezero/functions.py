from __future__ import annotations
from typing import Optional, Tuple, Union
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
        self.shape: tuple = shape
        self.x_shape: Optional[tuple] = None  # 入力のshapeを記憶する

    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x: np.ndarray = xs[0]
        self.x_shape: Optional[tuple] = x.shape
        y: np.ndarray = x.reshape(self.shape)
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


class BroadCastTo(Function):
    def __init__(self, shape: tuple, name=None) -> None:
        super().__init__(name)
        self.shape = shape

    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        y: np.ndarray = np.broadcast_to(x, self.shape)
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy: Variable = gys[0]

        return sum_to(gy)


def broad_cast_to(x: Variable, shape: tuple, name="") -> Variable:
    y = BroadCastTo(shape, name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class SumTo(Function):
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x: np.ndarray = xs[0]
        self.__x_shape: tuple = x.shape
        y = x.sum(axis=0)
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy: Variable = gys[0]
        y: Variable = broad_cast_to(gy, self.__x_shape)
        return y


def sum_to(x: Variable, name="") -> Variable:
    y = SumTo(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class MatMul(Function):
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 2:
            raise ValueError
        x = xs[0]
        W = xs[1]
        # print(type(x), type(W))
        # print(x)
        y = x.dot(W)
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy: Variable = gys[0]
        x, W = self.inputs
        # print(type(gy), type(x), type(W))
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return [gx, gW]


def matmul(x: Variable, W: Variable, name=""):
    return MatMul(name)(x, W)

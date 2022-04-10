from __future__ import annotations
import collections
from typing import List, Tuple, Union
import numpy as np
import dezero.utils as utils

import dezero
import abc


class Function(abc.ABC):
    def __init__(self, name=None) -> None:
        self.inputs: Tuple[Variable] = tuple()
        self.outputs: Tuple[Variable] = tuple()
        self.name = name

    @abc.abstractmethod
    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        return NotImplementedError

    @abc.abstractmethod
    def backward(self, gy: np.ndarray) -> Union[Tuple[np.ndarray], np.ndarray]:
        return NotImplementedError

    def __call__(self, *inputs: Variable) -> Union[Tuple[Variable], Variable]:
        self.inputs = [self._as_variable(input) for input in inputs]

        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = tuple([Variable(utils.as_array(y)) for y in ys])

        self.outputs = outputs
        for output in self.outputs:
            output.creator = self

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def _as_variable(self, obj) -> Variable:
        if isinstance(obj, Variable):
            return obj
        return Variable(obj)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        return np.exp(self.inputs[0].data) * gy


def exp(x: Variable, name="") -> Variable:
    y = Exp(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Add(Function):
    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 2:
            raise ValueError
        x0, x1 = xs[0], xs[1]
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        return (gy, gy)


def add(x1: Variable, x2: Variable, name="") -> Variable:
    y = Add(name)(x1, x2)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Neg(Function):
    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        y = -x
        return y

    def backward(self, gy: np.ndarray) -> Union[Tuple[np.ndarray], np.ndarray]:
        return -gy


def neg(x: Variable, name="") -> Variable:
    y = Neg(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Mul(Function):
    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 2:
            raise ValueError
        x0, x1 = xs[0], xs[1]
        y = x0 * x1
        return y

    def backward(self, gy: np.ndarray) -> Tuple[np.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return (x1 * gy, x0 * gy)


def mul(x1: Variable, x2: Variable, name="") -> Variable:
    y = Mul(name)(x1, x2)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Sub(Function):
    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 2:
            raise ValueError
        x0, x1 = xs[0], xs[1]
        y = x0 - x1
        return y

    def backward(self, gy: np.ndarray) -> Union[Tuple[np.ndarray], np.ndarray]:
        return (gy, -gy)


def sub(x1: Variable, x2: Variable, name="") -> Variable:
    y = Sub(name)(x1, x2)
    if not isinstance(y, Variable):
        raise TypeError
    return y


def rsub(x1: Variable, x2: Variable, name="") -> Variable:
    y = Sub(name)(x2, x1)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Div(Function):
    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        if len(xs) != 2:
            raise ValueError
        x0, x1 = xs[0], xs[1]
        y = x0 / x1
        return y

    def backward(self, gy: np.ndarray) -> Union[Tuple[np.ndarray], np.ndarray]:
        x1, x2 = self.inputs[0].data, self.inputs[1].data
        return (1 / x2, -x1 / x2 ** 2)


def div(x1: Variable, x2: Variable, name="") -> Variable:
    y = Div(name)(x1, x2)
    if not isinstance(y, Variable):
        raise TypeError
    return y


def rdiv(x1: Variable, x2: Variable, name="") -> Variable:
    y = Div(name)(x2, x1)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Pow(Function):
    def __init__(self, c, name=None):
        self.c = c
        self.name = name

    def forward(self, *xs: np.ndarray) -> Union[Tuple, np.ndarray]:
        x = xs[0]
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x: Variable, c: int, name="") -> Variable:
    y = Pow(c, name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Variable:
    def __init__(self, data: np.ndarray, name=None) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.grad = None
        self.creator: Union[Function, None] = None
        self.name = name

    def backward(self):
        self.grad = np.ones_like(self.data)

        funcs = collections.deque()
        funcs.append(self.creator)
        while funcs:
            func: Function = funcs.popleft()
            # print(func.name)
            gys = [y.grad for y in func.outputs]

            gxs = utils.as_array(func.backward(*gys))
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    if x.creator not in funcs:
                        funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()

    def transpose(self):
        return Variable(self.data.T)

    def reshape(self, shape: tuple):
        return dezero.functions.reshape(self, shape)

    def __len__(self):
        return len(self.data)

    def __add__(self, other: Union[Variable, np.ndarray, float, int]) -> Variable:
        if not isinstance(other, Variable):
            other = Variable(utils.as_array(other))
        return add(self, other)

    def __radd__(self, other: Union[np.ndarray, float, int]) -> Variable:
        return add(self, Variable(utils.as_array(other)))

    def __mul__(self, other: Union[Variable, np.ndarray, float, int]) -> Variable:
        if not isinstance(other, Variable):
            other = Variable(utils.as_array(other))
        return mul(self, other)

    def __rmul__(self, other: Union[np.ndarray, float, int]) -> Variable:
        return mul(self, Variable(utils.as_array(other)))

    def __neg__(self) -> Variable:
        return neg(self)

    def __sub__(self, other: Union[Variable, float, int]) -> Variable:
        if not isinstance(other, Variable):
            other = Variable(utils.as_array(other))
        return sub(self, other)

    def __rsub__(self, other: Union[np.ndarray, float, int]) -> Variable:
        return rsub(self, Variable(utils.as_array(other)))

    def __truediv__(self, other: Union[Variable, np.ndarray, float, int]) -> Variable:
        if not isinstance(other, Variable):
            other = Variable(utils.as_array(other))
        return div(self, other)

    def __rtruediv__(self, other: Union[np.ndarray, float, int]) -> Variable:
        return rdiv(self, Variable(utils.as_array(other)))

    def __pow__(self, c: int) -> Variable:
        return pow(self, c)

    def __repr__(self) -> str:
        ret = ""
        ret += "Variable("
        ret += str(self.data).replace("\n", "\n" + " " * 9)
        ret += ")"
        return ret

from __future__ import annotations
import abc
import collections
from typing import Optional, Union
import numpy as np

import dezero


class Function(abc.ABC):
    def __init__(self, name=None) -> None:
        self.inputs: list[Variable] = []
        self.outputs: list[Variable] = []
        self.__name = name

    @abc.abstractmethod
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        return NotImplementedError

    @abc.abstractmethod
    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        return NotImplementedError

    @property
    def name(self):
        if self.__name is None:
            return str(self)
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = value

    def __call__(self, *inputs: Variable) -> Union[list[Variable], Variable]:
        self.inputs = [self._as_variable(input) for input in inputs]

        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(dezero.utils.as_array(y)) for y in ys]

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
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        return np.exp(x)

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        return np.exp(self.inputs[0].data) * gy


def exp(x: Variable, name="") -> Variable:
    y = Exp(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Add(Function):
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 2:
            raise ValueError
        x0, x1 = xs[0], xs[1]
        y = x0 + x1
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        return [gy, gy]


def add(x1: Variable, x2: Variable, name="") -> Variable:
    y = Add(name)(x1, x2)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Neg(Function):
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 1:
            raise ValueError
        x = xs[0]
        y = -x
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        return -gy


def neg(x: Variable, name="") -> Variable:
    y = Neg(name)(x)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Mul(Function):
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 2:
            raise ValueError
        x0, x1 = xs[0], xs[1]
        y = x0 * x1
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy: Variable = gys[0]
        x0: Variable
        x1: Variable
        x0, x1 = self.inputs[0], self.inputs[1]
        return [x1 * gy, x0 * gy]


def mul(x1: Variable, x2: Variable, name="") -> Variable:
    y = Mul(name)(x1, x2)
    if not isinstance(y, Variable):
        raise TypeError
    return y


class Sub(Function):
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 2:
            raise ValueError
        x0, x1 = xs[0], xs[1]
        y = x0 - x1
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        return [gy, -gy]


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
    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        if len(xs) != 2:
            raise ValueError
        x0, x1 = xs[0], xs[1]
        y = x0 / x1
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        x0, x1 = self.inputs[0], self.inputs[1]
        return [gy / x1, -x0 * gy / x1 ** 2]


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
    def __init__(self, c: int, name=None):
        self.c = c
        super().__init__(name)

    def forward(self, *xs: np.ndarray) -> Union[list[np.ndarray], np.ndarray]:
        x = xs[0]
        y = x ** self.c
        return y

    def backward(self, *gys: Variable) -> Union[list[Variable], Variable]:
        if len(gys) != 1:
            raise ValueError
        gy = gys[0]
        x = self.inputs[0]
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
        self.grad: Optional[Variable] = None
        self.creator: Optional[Function] = None
        self.__name: Optional[str] = name

    def backward(self, display_fname=False) -> None:
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = collections.deque()
        funcs.append(self.creator)
        while funcs:
            func: Function = funcs.popleft()
            if display_fname:
                # print(func.name)
                print(func, func.outputs)
                # print(func.outputs)
            gys: list[Variable] = []
            for y in func.outputs:
                if y.grad is None:
                    raise ValueError
                gys.append(y.grad)

            # gxs = dezero.utils.as_array(func.backward(*gys))
            gxs = func.backward(*gys)
            if not isinstance(gxs, list):
                gxs = [gxs]

            # 逆伝播の結果を対応する入力変数に充てがう処理
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
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return self.transpose()

    @property
    def name(self):
        if self.__name is None:
            return str(self)
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = value

    def transpose(self):
        return dezero.functions.transpose(self)

    def reshape(self, shape: tuple):
        return dezero.functions.reshape(self, shape)

    def __len__(self):
        return len(self.data)

    def __add__(self, other: Union[Variable, np.ndarray, float, int]) -> Variable:
        if not isinstance(other, Variable):
            other = Variable(dezero.utils.as_array(other))
        if self.shape != other.shape:
            if self.size > other.size:
                broadcasted_other = dezero.functions.broad_cast_to(other, self.shape)
                return add(self, broadcasted_other)
            else:
                broadcasted_self = dezero.functions.broad_cast_to(self, other.shape)
                return add(broadcasted_self, other)
        return add(self, other)

    def __radd__(self, other: Union[np.ndarray, float, int]) -> Variable:
        return add(self, Variable(dezero.utils.as_array(other)))

    def __mul__(self, other: Union[Variable, np.ndarray, float, int]) -> Variable:
        if not isinstance(other, Variable):
            other = Variable(dezero.utils.as_array(other))
        return mul(self, other)

    def __rmul__(self, other: Union[np.ndarray, float, int]) -> Variable:
        return mul(self, Variable(dezero.utils.as_array(other)))

    def __neg__(self) -> Variable:
        return neg(self)

    def __sub__(self, other: Union[Variable, float, int]) -> Variable:
        if not isinstance(other, Variable):
            other = Variable(dezero.utils.as_array(other))
        return sub(self, other)

    def __rsub__(self, other: Union[np.ndarray, float, int]) -> Variable:
        return rsub(self, Variable(dezero.utils.as_array(other)))

    def __truediv__(self, other: Union[Variable, np.ndarray, float, int]) -> Variable:
        if not isinstance(other, Variable):
            other = Variable(dezero.utils.as_array(other))
        return div(self, other)

    def __rtruediv__(self, other: Union[np.ndarray, float, int]) -> Variable:
        return rdiv(self, Variable(dezero.utils.as_array(other)))

    def __pow__(self, c: int) -> Variable:
        return pow(self, c)

    def __repr__(self) -> str:
        ret = ""
        ret += "Variable("
        ret += str(self.data).replace("\n", "\n" + " " * 9)
        ret += ")"
        return ret

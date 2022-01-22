import collections
from typing import Union, TYPE_CHECKING
import numpy as np
import dezero.utils as utils

from dezero.core_simple import mul

if TYPE_CHECKING:
    from dezero.core_simple import Function, mul


class Variable(object):
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

    def __len__(self):
        return len(self.data)

    # def __mul__(self, obj):
    #     return mul(self, obj)

    # def __repr__(self) -> str:
    #     txt = ""
    #     txt += "Variable("


Variable.__mul__ == mul

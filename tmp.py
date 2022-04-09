from os import name
import numpy as np
from dezero.core import Variable, pow, sub, mul

# x = Variable(np.array(2.0))
# y = x ** 4 - 2 * x ** 2
# y.backward()
# assert x.grad.data == np.array(24.0)

# gx = x.grad
# x.cleargrad()
# print("*********")
# print(gx)
# gx.backward()
# assert x.grad.data == np.array(68.0)

x = Variable(np.array(2.0))
a = pow(x, 4, name="f1")
b = pow(x, 2, name="f2")
c = mul(b, Variable(np.array(2.0)), name="f3")
y = sub(a, c, name="f4")

y.backward(display_fname=True)
print("y: ", y.data)
print("gy: ", y.grad)
print("x.grad: ", x.grad.data)

gx = x.grad
gx.backward(display_fname=True)
print("x.grad: ", x.data)

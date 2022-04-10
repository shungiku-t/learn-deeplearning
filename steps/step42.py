"""線形回帰の実験"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import numpy as np
from dezero.core import Variable
from dezero.functions import matmul, sum_to
from dezero.models import linear_simple

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)

x, y = Variable(x), Variable(y)


W: Variable = Variable(np.zeros((1, 1)))
b: Variable = Variable(np.zeros(1))


def predict(x) -> Variable:
    y = matmul(x, W) + b
    return y


def mean_squared_error(y0: Variable, y1: Variable) -> Variable:
    """平均二乗誤差"""
    diff = y0 - y1
    return sum_to(diff ** 2) / len(y0)


fig: Figure = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ims = []

itr = 50
lr = 0.1
for _ in range(itr):
    mse = mean_squared_error(y, linear_simple(x, W, b))
    # mse = mean_squared_error(y, predict(x))
    W.cleargrad()
    b.cleargrad()
    mse.backward()

    W.data = W.data - W.grad.data * lr
    b.data = b.data - b.grad.data * lr

    print(W, b, mse)
    ims.append(ax.plot(x.data, predict(x).data))

ax.scatter(x.data, y.data, c="b")

ani = animation.ArtistAnimation(fig, ims, interval=100)
plt.show()

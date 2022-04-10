import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import true

from dezero.core import Variable
from dezero.models import nuralnet_simple
from dezero.loss import mean_squared_error

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y, c="b")

I = 1
H = 10
O = 1

# W1: Variable = Variable(np.zeros((1, H)))
# b1: Variable = Variable(np.zeros(H))
# W2: Variable = Variable(np.zeros((H, O)))
# b2: Variable = Variable(np.zeros(O))
W1: Variable = Variable(0.01 * np.random.randn(I, H))
b1: Variable = Variable(np.zeros(H))
W2: Variable = Variable(0.01 * np.random.randn(H, O))
b2: Variable = Variable(np.zeros(O))

iters = 10000
lr = 0.2

for _ in range(iters):
    W1.cleargrad()
    W2.cleargrad()
    b1.cleargrad()
    b1.cleargrad()

    y_pred: Variable = nuralnet_simple(x, W1, W2, b1, b2)
    loss = mean_squared_error(Variable(y), y_pred)
    loss.backward()

    W1.data -= W1.grad.data * lr
    W2.data -= W2.grad.data * lr
    b1.data -= b1.grad.data * lr
    b2.data -= b2.grad.data * lr

    # print(W1, W2, b1, b2)
    print(loss)

sorted_x = np.sort(x, axis=0)
ax.plot(sorted_x, nuralnet_simple(sorted_x, W1, W2, b1, b2).data)
plt.show()

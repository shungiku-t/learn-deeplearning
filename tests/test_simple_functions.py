import numpy as np
from dezero.core_simple import Function, Variable
import dezero.functions as F


class TestReshape:
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    # gy = np.array([4,5,6,1,2,3])

    def test_forward(self):
        expected = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = F.reshape(self.x, shape=(6,))

        assert (y.data == expected.data).all()

    def test_backward(self):
        expected = np.array([[1, 1, 1], [1, 1, 1]])

        y = F.reshape(self.x, shape=(6,))
        y.backward()
        assert (self.x.grad == expected).all()
        self.x.cleargrad()

import numpy as np
from dezero.core import Variable


class TestVariable:

    v0: Variable = Variable(np.array(2.0))
    v1: Variable = Variable(np.array(3.0))
    n0: np.ndarray = np.array(2.0)
    n1: np.ndarray = np.array(3.0)
    f0: float = 2.0
    f1: float = 3.0
    i0: int = 2
    i1: int = 3

    def test_shape(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        expected = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = x.reshape(shape=(6,))
        assert (y.data == expected.data).all()

    def test_transpose(self):
        expected = Variable(np.array([[1, 4], [2, 5], [3, 6]]))
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.transpose()

        assert (y.data == expected.data).all()

        y = x.T
        assert (y.data == expected.data).all()

    def test_add_forward(self):
        expected = Variable(np.array(5.0))

        # Varibleのみ
        y = self.v0 + self.v1
        assert y.data == expected.data

        # numpy
        y = self.v0 + self.n1
        assert y.data == expected.data

        y = self.n0 + self.v1
        assert y.data == expected.data

        # float
        y = self.v0 + self.f1
        assert y.data == expected.data

        y = self.f0 + self.v1
        assert y.data == expected.data

        # int
        y = self.v0 + self.i1
        assert y.data == expected.data

        y = self.i0 + self.v1
        assert y.data == expected.data

    def test_sub(self):
        expected = Variable(np.array(-1.0))

        # Varibleのみ
        y = self.v0 - self.v1
        assert y.data == expected.data

        # numpy
        y = self.v0 - self.n1
        assert y.data == expected.data

        y = self.n0 - self.v1
        assert y.data == expected.data

        # float
        y = self.v0 - self.f1
        assert y.data == expected.data

        y = self.f0 - self.v1
        assert y.data == expected.data

        # int
        y = self.v0 - self.i1
        assert y.data == expected.data

        y = self.i0 - self.v1
        assert y.data == expected.data

    def test_mul(self):
        expected = Variable(np.array(6.0))

        # Varibleのみ
        y = self.v0 * self.v1
        assert y.data == expected.data

        # numpy
        y = self.v0 * self.n1
        assert y.data == expected.data

        y = self.n0 * self.v1
        assert y.data == expected.data

        # float
        y = self.v0 * self.f1
        assert y.data == expected.data

        y = self.f0 * self.v1
        assert y.data == expected.data

        # int
        y = self.v0 * self.i1
        assert y.data == expected.data

        y = self.i0 * self.v1
        assert y.data == expected.data

    def test_div(self):
        expected = Variable(np.array(2.0 / 3.0))

        # Varibleのみ
        y = self.v0 / self.v1
        assert y.data == expected.data

        # numpy
        y = self.v0 / self.n1
        assert y.data == expected.data

        y = self.n0 / self.v1
        assert y.data == expected.data

        # float
        y = self.v0 / self.f1
        assert y.data == expected.data

        y = self.f0 / self.v1
        assert y.data == expected.data

        # int
        y = self.v0 / self.i1
        assert y.data == expected.data

        y = self.i0 / self.v1
        assert y.data == expected.data

    def test_neg(self):
        expected = Variable(np.array(-2.0))
        # Varibleのみ
        y = -self.v0
        assert y.data == expected.data

    def test_pow(self):
        expected = np.array(4.0)

        y = self.v0 ** 2
        assert y.data == expected.data

    # def test_broadcast(self):
    #     expected: Variable = Variable(np.array([[1,2,3],[1,2,3]]))

    #     x: Variable = Variable(np.array([1,2,3]))

    def test_backward(self):
        x = Variable(np.array(2.0))
        y = x ** 4 - 2 * x ** 2
        y.backward()
        assert x.grad.data == np.array(24.0)

        gx = x.grad
        x.cleargrad()
        print("*********")
        print(gx)
        gx.backward()
        assert x.grad.data == np.array(44.0)

    # def test_matyas(self):
    #     x = Variable(np.array(1.0))
    #     y = Variable(np.array(1.0))
    #     z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    #     z.backward()
    #     print(x.grad)
    #     print(y.grad)

from unittest import expectedFailure
import numpy as np
from dezero.core_simple import Variable


class TestVariable:

    v0 = Variable(np.array(2.0))
    v1 = Variable(np.array(3.0))
    n0 = np.array(2.0)
    n1 = np.array(3.0)
    f0 = 2.0
    f1 = 3.0
    i0 = 2
    i1 = 3

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

    def test_add(self):
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

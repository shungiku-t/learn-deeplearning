import numpy as np
from dezero.core import Variable
from dezero.functions import (
    BroadCastTo,
    MatMul,
    Reshape,
    SumTo,
    broad_cast_to,
    matmul,
    reshape,
    sum_to,
)


class TestReshape:
    def test_forward(self):
        expected: np.ndarray = np.array([[1, 2], [3, 4], [5, 6]])
        x: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
        f: Reshape = Reshape((3, 2))
        y: np.ndarray = f.forward((x))
        assert (y == expected).all()

    def test_backward(self):
        expected: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
        gy: np.ndarray = np.array([[1, 2], [3, 4], [5, 6]])
        f: Reshape = Reshape((3, 2))
        f.x_shape: tuple = (2, 3)
        gx: np.ndarray = f.backward(gy)
        assert (gx == expected).all()

    def test_reshape(self):
        expected: Variable = Variable(np.array([[1, 2], [3, 4], [5, 6]]))
        x: Variable = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y: Variable = reshape(x, shape=(3, 2))
        assert (y.data == expected.data).all()


class TestBroadCastTo:
    def test_forward(self):
        expected: np.ndarray = np.array([[1, 2, 3], [1, 2, 3]])

        x: np.ndarray = np.array([1, 2, 3])
        f = BroadCastTo((2, 3))
        y: np.ndarray = f.forward((x))
        assert np.array_equal(y, expected)

    def test_backward(self):
        expected_forward: Variable = Variable(np.array([[1, 2, 3], [1, 2, 3]]))
        expected_backward: Variable = Variable(np.array([2, 2, 2]))

        # まずはforward
        x: Variable = Variable(np.array([1, 2, 3]))
        y: Variable = BroadCastTo((2, 3))(x)
        assert np.array_equal(y.data, expected_forward.data)

        # 次にbackward
        y.backward()
        assert np.array_equal(x.grad.data, expected_backward.data)

    def test_broad_cast_to(self):
        expected: Variable = Variable(np.array([[1, 2, 3], [1, 2, 3]]))
        x: Variable = Variable(np.array([1, 2, 3]))
        y: Variable = broad_cast_to(x, shape=(2, 3))
        assert np.array_equal(y.data, expected.data)


class TestSumTo:
    def test_forward(self):
        expected: np.ndarray = np.array([5, 7, 9])
        x: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
        f: SumTo = SumTo()
        y: np.ndarray = f.forward((x))
        assert (y == expected).all()

    def test_backward(self):
        expected_forward: Variable = Variable(np.array([5, 7, 9]))
        expected_backward: Variable = Variable(np.array([[1, 1, 1], [1, 1, 1]]))

        # まずはforward
        x: Variable = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y: Variable = SumTo()(x)
        assert np.array_equal(y.data, expected_forward.data)

        # 次にbackward
        y.backward()
        assert np.array_equal(x.grad.data, expected_backward.data)

    def test_sum_to(self):
        expected: Variable = Variable(np.array([5, 7, 9]))
        x: Variable = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y: Variable = sum_to(x)
        assert np.array_equal(y.data, expected.data)


class TestMatMul:
    def test_forward(self):
        expected = np.array([[19, 22], [43, 50]])
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        f: MatMul = MatMul()
        y = f.forward(*(a, b))
        assert np.array_equal(y, expected)

    def test_backward(self):
        x = Variable(np.random.randn(2, 3))
        W = Variable(np.random.randn(3, 4))
        y = matmul(x, W)
        y.backward()
        assert x.grad.shape == (2, 3)
        assert W.grad.shape == (3, 4)

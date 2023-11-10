import numpy as np

import matrix_functions as mf


def test_cheb_nodes():
    assert np.allclose(mf.cheb_nodes(2), np.array([1, 0, -1]) * np.sqrt(3) / 2)


def test_cheb_interpolation():
    deg = 5
    poly_coeff = np.random.default_rng(42).standard_normal(size=deg + 1)

    def polynomial(x):
        return (np.vander(np.atleast_1d(x), N=deg + 1) @ poly_coeff).reshape(
            np.shape(x)
        )

    a = 0.5
    b = 2.5
    interpolant = mf.cheb_interpolation(deg, polynomial, a, b)
    xs = np.linspace(a, b)
    # scalar version
    assert np.allclose([interpolant(x) for x in xs], polynomial(xs))
    # vector version
    assert np.allclose(interpolant(xs), polynomial(xs))

    # evaluation points are the same as the interpolation points
    cheb_nodes = mf.cheb_nodes(deg, a, b)
    assert np.allclose(interpolant(cheb_nodes[0]), polynomial(cheb_nodes[0]))
    assert np.allclose(interpolant(cheb_nodes), polynomial(cheb_nodes))

    const_interpolant = mf.cheb_interpolation(0, polynomial, a, b)
    const_interp_node = mf.cheb_nodes(0, a, b)
    assert np.allclose(const_interpolant(xs), polynomial(const_interp_node))
    assert np.allclose(
        const_interpolant(const_interp_node), polynomial(const_interp_node)
    )


def test_cheb_vandermonde():
    x = np.array([-1, -0.5, 0, 0.5, 1])
    reference = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [1.0, -0.5, -1.0, -0.5, 1.0],
            [-1.0, 1.0, 0.0, -1.0, 1.0],
            [1.0, -0.5, 1.0, -0.5, 1.0],
        ]
    ).T
    assert np.allclose(mf.cheb_vandermonde(x, 4), reference)

    # The values at each node are the same even if the nodes are linearly transformed
    assert np.allclose(mf.cheb_vandermonde(2 * x + 10, 4), reference)

    # Unless we specify the interval
    scaled_reference = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-2.0, -1, 0.0, 1, 2.0],
            [7.0, 1.0, -1.0, 1.0, 7],
            [-26.0, -1.0, 0.0, 1.0, 26.0],
            [97.0, 1.0, 1.0, 1.0, 97.0],
        ]
    ).T
    assert np.allclose(
        mf.cheb_vandermonde(2 * x, 4, interval=(-1, 1)), scaled_reference
    )


def test_cheb_regression_errors():
    points = mf.utils.linspace(3, 5, num=3)

    def f(x):
        return (x - 4) ** 2

    assert np.allclose(
        mf.chebyshev.cheb_regression_errors(0, f(points), points), f(points) - (2 / 3)
    )
    assert np.allclose(
        mf.chebyshev.cheb_regression_errors(1, f(points), points), f(points) - (2 / 3)
    )
    assert np.allclose(mf.chebyshev.cheb_regression_errors(2, f(points), points), 0)

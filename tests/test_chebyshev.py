import numpy as np

import matrix_functions as mf


def test_cheb_nodes():
    assert np.allclose(mf.cheb_nodes(2), np.array([1, 0, -1]) * np.sqrt(3)/2)


def test_cheb_interpolation():
    deg = 5
    poly_coeff = np.random.default_rng(42).standard_normal(size=deg+1)

    def polynomial(x):
        return (np.vander(np.atleast_1d(x), N=deg+1) @ poly_coeff).reshape(np.shape(x))

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
    assert np.allclose(const_interpolant(const_interp_node), polynomial(const_interp_node))


def test_cheb_vandermonde():
    x = np.array([-1, -0.5, 0, 0.5, 1])
    M = mf.cheb_vandermonde(x, 4)
    reference = np.array([
        [1., 1., 1., 1., 1.],
        [-1., -0.5, 0., 0.5, 1.],
        [1., -0.5, -1., -0.5, 1.],
        [-1., 1., 0., -1., 1.],
        [1., -0.5, 1., -0.5, 1.]
    ]).T
    assert np.allclose(M, reference)

    # The values at each node are the same even if the nodes are linearly transformed
    assert np.allclose(mf.cheb_vandermonde(2 * x + 10, 4), reference)

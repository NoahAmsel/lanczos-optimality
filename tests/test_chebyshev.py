import numpy as np

import matrix_functions as mf


def test_cheb_nodes():
    assert np.allclose(mf.cheb_nodes(2), np.array([1, 0, -1]) * np.sqrt(3)/2)


def test_cheb_interpolation():
    deg = 5
    poly_coeff = np.random.default_rng(42).standard_normal(size=deg+1)

    def polynomial(x):
        return np.vander(x, N=deg+1) @ poly_coeff

    a = 0.5
    b = 2.5
    interpolant = mf.cheb_interpolation(deg, polynomial, a, b)
    xs = np.linspace(a, b)
    assert np.allclose([interpolant(x) for x in xs], polynomial(xs))

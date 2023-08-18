import flamp
import numpy as np

import matrix_functions as mf


def test_discrete_remez_error():
    a = 3
    b = 4
    points = mf.utils.linspace(a, b, 101)
    def inv_sqrt(x): return x**(-1/2)

    for degree in range(10):
        remez_error_continuous = mf.remez_error(degree, inv_sqrt, domain=(a, b), max_iter=10, tol=1e-16)
        remez_error_discrete = mf.discrete_remez_error(degree, inv_sqrt(points), points, max_iter=10, tol=1e-16)
        regression_error = mf.norm(mf.chebyshev.cheb_regression_errors(degree, inv_sqrt(points), points), ord=np.inf)
        assert remez_error_continuous < regression_error
        assert remez_error_discrete < regression_error


def test_oscillations():
    points = mf.utils.linspace(-np.pi, np.pi, 101, dtype=np.dtype('O'))
    remez_error_discrete = mf.discrete_remez_error(0, flamp.cos(points), points, max_iter=10, tol=1e-16)
    assert np.abs(remez_error_discrete - 1) < flamp.gmpy2.mpfr('1e-17')

    # what if 
    points = flamp.to_mp([-2, -1, 0, 1, 2])
    vals = flamp.to_mp([-1, 1, 0, 1, -1])
    remez_error_discrete = mf.discrete_remez_error(0, vals, points, max_iter=10, tol=1e-16)
    assert np.abs(remez_error_discrete - 1) < flamp.gmpy2.mpfr('1e-17')

    points = flamp.to_mp([-3, -2, -1, 0, 1, 2, 3])
    vals = flamp.to_mp([-1, 1, 0, 0, 0, 1, -1])
    remez_error_discrete = mf.discrete_remez_error(0, vals, points, max_iter=10, tol=1e-16)
    assert np.abs(remez_error_discrete - 1) < flamp.gmpy2.mpfr('1e-17')

def test_perfect():
    points = flamp.to_mp([-1, 0, 1])
    vals = flamp.to_mp([0, 0, 0])
    remez_error = mf.discrete_remez_error(0, vals, points, max_iter=10, tol=1e-16)
    assert np.abs(remez_error) < flamp.gmpy2.mpfr('1e-17')



# TODO add test case where, with initial (degree 0) guess, there are 4 local extrema.
# like f(x) = (x+1)(x)(x-1) over [-2,2]
# or sin(x)
# also try a test case with this triangle wave and degree 0
# np.array([0, 1, 0, 1, 0]) (((where the initial points are the first and last?)))
# p. 301 of https://dl.acm.org/doi/pdf/10.1145/321281.321282
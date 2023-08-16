import numpy as np

import matrix_functions as mf


def test_discrete_remez_error():
    a = 3
    b = 4
    points = mf.utils.linspace(a, b, 101)
    def inv_sqrt(x): return x**(-1/2)

    for degree in range(10):
        remez_error_continuous = mf.remez_error(degree, inv_sqrt, domain=(a, b), max_iter=10, tol=1e-16)
        remez_error_discrete = mf.discrete_remez_error(degree, inv_sqrt, points, max_iter=10, tol=1e-16)
        regression_error = mf.norm(mf.chebyshev.cheb_regression_errors(degree, inv_sqrt(points), points), ord=np.inf)
        assert remez_error_continuous < regression_error
        assert remez_error_discrete < regression_error

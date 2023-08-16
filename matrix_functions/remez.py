import warnings

import flamp
import numpy as np
from scipy.signal import find_peaks

from .chebyshev import cheb_nodes, cheb_vandermonde, cheb_regression_errors
from .utils import arange


def discrete_remez_error(degree, f, points, max_iter=100, tol=1e-10):
    points = np.array(points)
    a = points.min()
    b = points.max()
    def shift(x): return 2*(x-a)/(b-a) - 1  # send [a,b] to [-1, 1]
    def unshift(x): return x*(b-a)/2 + (a+b)/2  # send [-1, 1] to [a, b]
    def ff(x): return f(unshift(x))
    points = shift(points)
    f_points = ff(points)

    # initial guess: chebyshev l2 regression
    my_errors = cheb_regression_errors(degree, f_points, points)
    X = points[np.sort(np.hstack((0, find_peaks(my_errors)[0], find_peaks(-my_errors)[0], len(points)-1)))]

    # TODO: I added this block to handle the special case where the function is interpolated perfectly by Chebyshev
    # regression but I'm not sure it's the right thing
    if (len(X) < degree + 2) and (np.max(np.abs(my_errors)) < tol):
        return 0, np.max(np.abs(my_errors)), X

    for _ in range(max_iter):
        V = np.hstack((
            cheb_vandermonde(X, degree), ((-1)**arange(degree+2, dtype=points.dtype))[:, np.newaxis]
        ))

        fX = ff(X)

        if points.dtype == np.dtype('O'):
            c = flamp.lu_solve(V, fX)
        else:
            c = np.linalg.solve(V, fX)
        E = np.abs(c[-1])  # lower bound on infinity norm error

        p = np.polynomial.chebyshev.Chebyshev(c[:-1])

        # upper bounbds on infinity norm error
        err_fun = f_points - p(points)
        F = np.max(np.abs(err_fun))

        if np.abs(E-F) < tol:
            return F

        extrema_indices = np.sort(np.hstack((0, find_peaks(err_fun)[0], find_peaks(-err_fun)[0], len(err_fun)-1)))
        assert len(extrema_indices) == degree + 2
        X = points[extrema_indices]

    warnings.warn(f"Remez fail to converge after {max_iter} iterations.\nE={E}\nF={F}")
    return F


def remez_error(degree, f, domain=[-1, 1], max_iter=100, n_grid=2000, tol=1e-10):
    a, b = domain
    grid = cheb_nodes(n_grid-1, a=a, b=b, dtype=np.dtype('O'))
    return discrete_remez_error(degree, f, grid, max_iter=max_iter, tol=tol)

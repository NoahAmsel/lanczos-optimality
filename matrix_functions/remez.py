import warnings

import flamp
import numpy as np
from scipy.signal import find_peaks

from .chebyshev import cheb_nodes, cheb_vandermonde, cheb_regression_errors
from .utils import arange


def extrema_indices(x, num):
    # The indices must be returned in sorted order!
    # Divide x into intervals where the sign is all the same
    # Hack! deal with the places where x = 0. Perturb these so they fall strictly in one interval or the other
    zero_ixs = np.nonzero(x == 0)[0]
    eps = np.min(np.abs(x)) / 2
    # alternate the signs of the perturbation in case there are consecutive zeros
    x[zero_ixs] = eps * (-1) ** np.arange(len(zero_ixs))

    # start_indices contains the index of the first element in each of these intervals
    # x < 0 instead of np.sign(x) because if x is 0, we want a tie break
    switch_ixs = list(np.nonzero(np.diff(np.sign(x)))[0] + 1)
    start_ixs = [0] + switch_ixs
    end_ixs = switch_ixs + [len(x)]
    intervals = zip(start_ixs, end_ixs)
    # find the max within each each interval
    max_indices = [start_ix + np.argmax(np.abs(x[start_ix:end_ix])) for start_ix, end_ix in intervals]
    # there may be more extreme points than we need
    # keep the biggest ones, preserving alternating signs
    while len(max_indices) > num:
        if x[max_indices[0]] > x[max_indices[-1]]:
            max_indices.pop()
        else:
            max_indices.pop(0)
    assert len(max_indices) == num
    return max_indices


def discrete_remez_error(degree, f_points, points, max_iter=100, tol=1e-10):
    # See page 301 of https://dl.acm.org/doi/pdf/10.1145/321281.321282
    points = np.array(points)
    a = points.min()
    b = points.max()
    def shift(x): return 2*(x-a)/(b-a) - 1  # send [a,b] to [-1, 1]
    def unshift(x): return x*(b-a)/2 + (a+b)/2  # send [-1, 1] to [a, b]
    points = shift(points)

    # initial guess: chebyshev l2 regression
    err_fun = cheb_regression_errors(degree, f_points, points)
    extreme_ixs = extrema_indices(err_fun, degree + 2)

    # TODO: I added this block to handle the special case where the function is interpolated perfectly by Chebyshev
    # regression but I'm not sure it's the right thing
    if (len(extreme_ixs) < degree + 2) and (np.max(np.abs(err_fun)) < tol):
        return np.max(np.abs(err_fun))

    for _ in range(max_iter):
        V = np.hstack((
            cheb_vandermonde(points[extreme_ixs], degree), ((-1)**arange(degree+2, dtype=points.dtype))[:, np.newaxis]
        ))

        fX = f_points[extreme_ixs]

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

        extreme_ixs = extrema_indices(err_fun, degree + 2)

    warnings.warn(f"Remez fail to converge after {max_iter} iterations.\nE={E}\nF={F}")
    return F


def remez_error(degree, f, domain=[-1, 1], max_iter=100, n_grid=2000, tol=1e-10):
    a, b = domain
    grid = cheb_nodes(n_grid-1, a=a, b=b, dtype=np.dtype('O'))
    return discrete_remez_error(degree, f(grid), grid, max_iter=max_iter, tol=tol)

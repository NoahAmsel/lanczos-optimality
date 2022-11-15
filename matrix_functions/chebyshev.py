from itertools import cycle, islice

import numpy as np

import flamp


def cheb_nodes(degree, a=-1., b=1., dtype=float):
    if dtype == np.dtype('O'):
        return (a+b)/2 + ((b-a)/2) * flamp.cos(
            flamp.linspace(1, 2 * degree + 1, num=(degree + 1)) * flamp.gmpy2.const_pi() / (2 * degree + 2))
    else:
        return (a+b)/2 + ((b-a)/2) * np.cos(np.linspace(1, 2 * degree + 1, num=(degree + 1)) * np.pi / (2 * degree + 2))


def cheb_interpolation(degree, f, a, b, dtype=float):
    """https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
    """
    x = cheb_nodes(degree, a=a, b=b, dtype=dtype)
    y = f(x)
    if dtype == np.dtype('O'):
        w = flamp.sin(flamp.linspace(1, 2 * degree + 1, num=(degree + 1)) * flamp.gmpy2.const_pi() / (2 * degree + 2))
    else:
        w = np.sin(np.linspace(1, 2 * degree + 1, num=(degree + 1)) * np.pi / (2 * degree + 2))
    w *= np.array(list(islice(cycle((1, -1)), len(w))))

    def interpolant(z):
        z = np.atleast_1d(z)
        coeffs = np.equal.outer(z, x).astype(dtype)
        z_isnt_interpolation_point = ~coeffs.any(axis=1).astype(bool)
        coeffs[z_isnt_interpolation_point, :] = w / (np.subtract.outer(z[z_isnt_interpolation_point], x))
        return np.squeeze((coeffs @ y) / coeffs.sum(axis=1))

    return interpolant


def cheb_vandermonde(x, max_degree):
    a = x.min()
    b = x.max()
    rescaled_x = 2 * (x - (a+b)/2) / (b-a)
    k = max_degree + 1
    M = np.empty((len(rescaled_x), k), dtype=x.dtype)
    M[:, 0] = 1.
    if k == 1:
        return M
    M[:, 1] = rescaled_x
    for i in range(2, k):
        M[:, i] = 2 * rescaled_x * M[:, i-1] - M[:, i-2]
    return M

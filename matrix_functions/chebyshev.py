import numpy as np


def cheb_nodes(degree, a=-1., b=1.):
    return (a+b)/2 + ((b-a)/2) * np.cos(np.linspace(1, 2 * degree + 1, num=(degree + 1)) * np.pi / (2 * degree + 2))


def cheb_interpolation(degree, f, a, b):
    """https://people.maths.ox.ac.uk/trefethen/barycentric.pdf
    """
    x = cheb_nodes(degree, a=a, b=b)
    y = f(x)
    w = np.sin(np.linspace(1, 2 * degree + 1, num=(degree + 1)) * np.pi / (2 * degree + 2))
    w *= np.array([1, -1] * degree)[:(degree+1)]

    def interpolant(z):
        coeffs = w / (z - x)
        return (coeffs @ y) / coeffs.sum()

    return interpolant

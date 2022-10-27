import numpy as np
import numpy.linalg as lin
import scipy.linalg

import flamp


def norm(x, ord=2):
    if (x.dtype == np.dtype('O')) and (ord == 2):
        return flamp.vector_norm(x)
    else:
        return lin.norm(x, ord=ord)


def eigh_tridiagonal(d, e):
    if np.result_type(d, e) == np.dtype('O'):
        # flamp.eigen_symmetric.tridiag_eigen modifies arrays in place, so make copies
        # it also expects e to be the same length as d, not one shorter like scipy does
        d = d.copy()
        e_long = flamp.empty(len(d))
        e_long[:len(e_long)-1] = e
        e_long[-1] = 0
        z = flamp.eye(len(d))
        flamp.eigen_symmetric.tridiag_eigen(flamp.gmpy2, d, e_long, z)
        return d, z
    else:
        return scipy.linalg.eigh_tridiagonal(d, e)


def generate_orthonormal(n, seed=None):
    M = np.random.default_rng(seed).normal(size=(n, n))
    Q, _ = lin.qr(M)
    return Q


def generate_symmetric(eigenvalues, seed=None):
    Q = generate_orthonormal(len(eigenvalues), seed=seed)
    return Q @ np.diag(eigenvalues) @ Q.T


def generate_model_spectrum(n, kappa, rho, lambda_1=1., seed=None):
    assert 0 < rho < 1
    lambda_n = kappa * lambda_1
    gap = lambda_n - lambda_1
    return lambda_1 + (np.linspace(0, n-1, num=n) / (n-1)) * gap * (rho ** np.linspace(n-1, 0, num=n))


def generate_outlier_spectrum(n, kappa, lambda_1=1.):
    eigenvalues = np.full(n, lambda_1)
    eigenvalues[-1] = kappa * lambda_1
    return eigenvalues

import numpy as np
import numpy.linalg as lin
import scipy.linalg
import torch

import flamp


def norm(x, ord=2):
    if (x.dtype == np.dtype('O')) and (ord == 2):
        return flamp.vector_norm(x)
    else:
        return lin.norm(x, ord=ord)


def linspace(start, stop, num=50, endpoint=True, dtype=None):
    if (dtype == np.dtype('O')) or ((dtype is None) and (np.result_type(start, stop) == np.dtype('O'))):
        if num == 1:
            # there's a bug in flamp.linspace, see https://github.com/c-f-h/flamp/issues/1.
            # so we have to do this workaround
            return flamp.to_mp(np.array([start]))

        return flamp.linspace(start, stop, num=num, endpoint=endpoint)
    else:
        return np.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)


def eigh(X):
    if type(X) == torch.Tensor:
        return torch.linalg.eigh(X)
    else:
        return lin.eigh(X)


def eigh_tridiagonal(d, e):
    if type(d) == torch.Tensor:
        mat = torch.diag(d) + torch.diag(e, 1) + torch.diag(e, -1)
        return torch.linalg.eigh(mat)
    elif np.result_type(d, e) == np.dtype('O'):
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


def tridiagonal(alpha, beta):
    return scipy.sparse.diags([beta, alpha, beta], [-1, 0, 1])


def generate_orthonormal(n, seed=None):
    M = np.random.default_rng(seed).normal(size=(n, n))
    Q, _ = lin.qr(M)
    return Q


def generate_symmetric(eigenvalues, seed=None):
    Q = generate_orthonormal(len(eigenvalues), seed=seed)
    return Q @ np.diag(eigenvalues) @ Q.T


def generate_model_spectrum(n, kappa, rho, lambda_1=1.):
    assert 0 < rho < 1
    lambda_n = kappa * lambda_1
    gap = lambda_n - lambda_1
    return lambda_1 + (np.linspace(0, n-1, num=n) / (n-1)) * gap * (rho ** np.linspace(n-1, 0, num=n))


def generate_outlier_spectrum(n, kappa, lambda_1=1.):
    eigenvalues = np.full(n, lambda_1)
    eigenvalues[-1] = kappa * lambda_1
    return eigenvalues


class DiagonalMatrix:
    def __init__(self, diag):
        self.diag = diag

    def __len__(self):
        return len(self.diag)

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]
        # This looks complicated but this let's the same formula
        # work when other is a vector or a matrix
        return (other.T * self.diag).T

    @property
    def dtype(self):
        return self.diag.dtype

    @property
    def shape(self):
        return len(self), len(self)

import numpy as np
import numpy.linalg as lin
import scipy.linalg

import flamp


def exp(x):
    if x.dtype == np.dtype("O"):
        return flamp.exp(x)
    else:
        return np.exp(x)


def norm(x, ord=2):
    if (x.dtype == np.dtype("O")) and (ord == 2):
        return flamp.vector_norm(x)
    else:
        return lin.norm(x, ord=ord)


def log(x):
    if x.dtype == np.dtype("O"):
        return flamp.log(x)
    else:
        return np.log(x)


def arange(n, dtype=None):
    if dtype == np.dtype("O"):
        return flamp.to_mp(np.arange(n))
    else:
        return np.arange(n)


def linspace(start, stop, num=50, endpoint=True, dtype=None):
    if (dtype == np.dtype("O")) or (
        (dtype is None)
        and ((type(start) == flamp.gmpy2.mpfr) or (type(stop) == flamp.gmpy2.mpfr))
    ):
        if num == 1:
            # there's a bug in flamp.linspace, see https://github.com/c-f-h/flamp/issues/1.
            # so we have to do this workaround
            return flamp.to_mp(np.array([start]))

        return flamp.linspace(start, stop, num=num, endpoint=endpoint)
    else:
        return np.linspace(start, stop, num=num, endpoint=endpoint, dtype=dtype)


def eigh(X):
    if X.dtype == np.dtype("O"):
        return flamp.eigh(X)
    else:
        return lin.eigh(X)


def eigh_tridiagonal(d, e, eigvals_only=False):
    if np.result_type(d, e) == np.dtype("O"):
        # flamp.eigen_symmetric.tridiag_eigen modifies arrays in place, so make copies
        # it also expects e to be the same length as d, not one shorter like scipy does
        d = d.copy()
        e_long = flamp.empty(len(d))
        e_long[: len(e_long) - 1] = e
        e_long[-1] = 0
        if eigvals_only:
            flamp.eigen_symmetric.tridiag_eigen(flamp.gmpy2, d, e_long, None)
            return d
        else:
            z = flamp.eye(len(d))
            flamp.eigen_symmetric.tridiag_eigen(flamp.gmpy2, d, e_long, z)
            return d, z
    else:
        return scipy.linalg.eigh_tridiagonal(d, e, eigvals_only=eigvals_only)


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

    def __pow__(self, other):
        return DiagonalMatrix(self.diag**other)

    def __rmatmul__(self, other):
        return self.__matmul__(other.T).T

    @property
    def T(self):
        return self

    @property
    def dtype(self):
        return self.diag.dtype

    @property
    def shape(self):
        return len(self), len(self)

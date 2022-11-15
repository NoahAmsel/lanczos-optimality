import numpy.linalg as lin

from .lanczos import lanczos
from .utils import norm, eigh_tridiagonal


def naive_fa(f, A, x):
    l, V = lin.eigh(A)
    # A = V @ np.diag(l) @ V.T
    return V @ (f(l) * (V.T @ x))


def diagonal_fa(f, a_diag, x):
    return f(a_diag) * x


def lanczos_fa(f, A, x, k=None):
    """See Stability of Lanczos Method page 5
    """

    Q, (alpha, beta) = lanczos(A, x, k)

    # We don't need to construct T, but if we did it would be this (see utils.tridiagonal):
    # T = scipy.sparse.diags([beta, alpha, beta], [-1, 0, 1]) # .toarray()
    # A = Q @ T.toarray() @ Q.T ... approximately

    T_lambda, T_V = eigh_tridiagonal(alpha, beta)
    return norm(x) * (Q @ (T_V @ (f(T_lambda) * T_V[0, :])))


def lanczos_fa_multi_k(f, A, x, ks=None, reorthogonalize=False):
    if ks is None:
        ks = range(1, len(x)+1)
    ks = list(ks)

    Q, (alpha, beta) = lanczos(A, x, max(ks), reorthogonalize=reorthogonalize)

    for k in ks:
        T_lambda, T_V = eigh_tridiagonal(alpha[:k], beta[:k-1])
        yield norm(x) * (Q[:, :k] @ (T_V @ (f(T_lambda) * T_V[0, :])))

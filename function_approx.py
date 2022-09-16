import numpy as np
import numpy.linalg as lin
import scipy.linalg

from lanczos import lanczos

def naive_fa(f, A, x):
    l, V = lin.eigh(A)
    # A = V @ np.diag(l) @ V.T
    return V @ (f(l) * (V.T @ x))

def diagonal_fa(f, a_diag, x):
    return f(a_diag) * x

def lanczos_fa(f, A, x, k=None):
    """See Stability of Lanczos Method page 5
    """

    Q, alpha, beta = lanczos(A, x, k)

    # We don't need to construct T, but if we did it would be this:
    # T = scipy.sparse.diags([beta, alpha, beta], [-1, 0, 1]) # .toarray()
    # A = Q @ T.toarray() @ Q.T ... approximately

    T_lambda, T_V = scipy.linalg.eigh_tridiagonal(alpha, beta)
    return lin.norm(x) * (Q @ (T_V @ (f(T_lambda) * T_V[0, :])))

def lanczos_fa_multi_k(f, A, x, ks=None):
    n = len(x)
    assert A.shape[0] == A.shape[1] == n

    if ks is None:
        ks = range(1, n+1)

    ks = list(ks)

    Q, alpha, beta = lanczos(A, x, max(ks))

    for k in ks:
        T_lambda, T_V = scipy.linalg.eigh_tridiagonal(alpha[:k], beta[:k-1])
        yield lin.norm(x) * (Q[:, :k] @ (T_V @ (f(T_lambda) * T_V[0, :])))

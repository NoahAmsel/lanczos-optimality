from .lanczos_decomp import LanczosDecomposition
from .utils import eigh


def naive_fa(f, A, x):
    l, V = eigh(A)
    # A = V @ np.diag(l) @ V.T
    return V @ (f(l) * (V.T @ x))


def diagonal_fa(f, a_diag, x):
    return f(a_diag) * x


def lanczos_fa(f, A, x, k=None, reorthogonalize=False, beta_tol=0):
    return LanczosDecomposition.fit(
        A, x, k=k, reorthogonalize=reorthogonalize, beta_tol=beta_tol).apply_function_to_start(f)

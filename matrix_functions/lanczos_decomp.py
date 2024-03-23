from functools import cache

import numpy as np
import scipy.sparse

from .lanczos import lanczos
from .utils import norm, eigh_tridiagonal, qr_solve, zeros


class SymmetricTridiagonal:
    def __init__(self, main_diagonal, off_diagonal):
        self.main_diagonal = main_diagonal
        self.off_diagonal = off_diagonal

    def prefix(self, k):
        return SymmetricTridiagonal(self.main_diagonal[:k], self.off_diagonal[: k - 1])

    def to_sparse(self):
        return scipy.sparse.diags(
            [self.off_diagonal, self.main_diagonal, self.off_diagonal],
            [-1, 0, 1],
            dtype=self.main_diagonal.dtype
        )

    def to_dense(self):
        # Normally you can get this from the sparse representation
        # But for some reason that doesn't work when the entries are sparse
        k = len(self.main_diagonal)
        out = zeros((k, k), dtype=np.result_type(self.main_diagonal, self.off_diagonal))
        np.fill_diagonal(out, self.main_diagonal)
        out[list(range(k-1)), list(range(1, k))] = self.off_diagonal
        out[list(range(1, k)), list(range(k-1))] = self.off_diagonal
        return out

    @cache
    def eigendecomp(self):
        return eigh_tridiagonal(self.main_diagonal, self.off_diagonal)


class LanczosDecomposition:
    def __init__(self, Q, T, norm_start_vector, next_q):
        self.Q = Q
        self.T = T
        self.norm_start_vector = norm_start_vector
        self.next_q = next_q

    @classmethod
    def fit(cls, A, q_1, k=None, reorthogonalize=False, beta_tol=0):
        Q, (alpha, beta), next_q = lanczos(
            A, q_1, k=k, reorthogonalize=reorthogonalize, beta_tol=beta_tol
        )
        return cls(Q, SymmetricTridiagonal(alpha, beta), norm(q_1), next_q)

    @cache
    def prefix(self, k):
        next_q = self.next_q if k == self.Q.shape[1] else self.Q[:, k]
        return LanczosDecomposition(
            self.Q[:, :k], self.T.prefix(k), self.norm_start_vector, next_q
        )

    def apply_function_to_start(self, f):
        T_lambda, T_V = self.T.eigendecomp()
        return self.norm_start_vector * (self.Q @ (T_V @ (f(T_lambda) * T_V[0, :])))

    def ritz_values(self):
        return eigh_tridiagonal(
            self.T.main_diagonal, self.T.off_diagonal, eigvals_only=True
        )

    def cg(self):
        return self.apply_function_to_start(lambda x: 1/x)

    def minres(self):
        # This is inefficient but correct.
        # See Pleiss Appendix C
        k = self.Q.shape[1]
        tilde_T = zeros((k+1, k), dtype=self.Q.dtype)
        tilde_T[:k, :k] = self.T.to_dense()
        tilde_T[-1, -1] = norm(self.next_q)
        e1 = zeros(k+1, dtype=self.Q.dtype)
        e1[0] = 1
        c = qr_solve(tilde_T, e1)
        return self.norm_start_vector * self.Q @ c

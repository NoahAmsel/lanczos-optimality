from functools import cache

import scipy.sparse

from .lanczos import lanczos
from .utils import norm, eigh_tridiagonal


class SymmetricTridiagonal:
    def __init__(self, main_diagonal, off_diagonal):
        self.main_diagonal = main_diagonal
        self.off_diagonal = off_diagonal

    def prefix(self, k):
        return SymmetricTridiagonal(self.main_diagonal[:k], self.off_diagonal[: k - 1])

    def to_sparse(self):
        return scipy.sparse.diags(
            [self.off_diagonal, self.main_diagonal, self.off_diagonal], [-1, 0, 1]
        )

    @cache
    def eigendecomp(self):
        return eigh_tridiagonal(self.main_diagonal, self.off_diagonal)


class LanczosDecomposition:
    def __init__(self, Q, T, norm_start_vector):
        self.Q = Q
        self.T = T
        self.norm_start_vector = norm_start_vector

    @classmethod
    def fit(cls, A, q_1, k=None, reorthogonalize=False, beta_tol=0):
        Q, (alpha, beta) = lanczos(
            A, q_1, k=k, reorthogonalize=reorthogonalize, beta_tol=beta_tol
        )
        return cls(Q, SymmetricTridiagonal(alpha, beta), norm(q_1))

    @cache
    def prefix(self, k):
        return LanczosDecomposition(
            self.Q[:, :k], self.T.prefix(k), self.norm_start_vector
        )

    def apply_function_to_start(self, f):
        T_lambda, T_V = self.T.eigendecomp()
        return self.norm_start_vector * (self.Q @ (T_V @ (f(T_lambda) * T_V[0, :])))

    def ritz_values(self):
        return eigh_tridiagonal(
            self.T.main_diagonal, self.T.off_diagonal, eigvals_only=True
        )

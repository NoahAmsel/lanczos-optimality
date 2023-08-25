import numpy as np

import flamp

from .chebyshev import cheb_nodes, cheb_vandermonde
from .function_approx import diagonal_fa
from .lanczos_decomp import LanczosDecomposition
from .remez import remez_error, discrete_remez_error
from .utils import DiagonalMatrix, norm


class DiagonalFAProblem:
    def __init__(self, f, spectrum, b, cache_k=None):
        self.f = f
        self.spectrum = spectrum
        self.b = b
        self.cached_decomp = None
        if cache_k is not None:
            # Future calls to lanczos_decomp with k <= cache_k will be fast
            self.lanczos_decomp(cache_k)

    def ground_truth(self):
        return diagonal_fa(self.f, self.spectrum, self.b)

    def A(self):
        return DiagonalMatrix(self.spectrum)

    def sqrtA(self):
        return DiagonalMatrix(flamp.sqrt(self.spectrum))

    def dim(self):
        return len(self.spectrum)

    def kappa(self):
        return np.abs(self.spectrum.max() / self.spectrum.min())

    def lanczos_decomp(self, k):
        if (self.cached_decomp is None) or (k > self.cached_decomp.Q.shape[1]):
            self.cached_decomp = LanczosDecomposition.fit(self.A(), self.b, k, reorthogonalize=True)
        return self.cached_decomp.prefix(k)

    def Q(self, k):
        return self.lanczos_decomp(k).Q

    def lanczos_error(self, k, norm_matrix_sqrt=None):
        lanczos_estimate = self.lanczos_decomp(k).apply_function_to_start(self.f)
        error = lanczos_estimate - self.ground_truth()
        if norm_matrix_sqrt is None:
            return norm(error)
        else:
            return norm(norm_matrix_sqrt @ error)

    def instance_optimal_error(self, k, norm_matrix_sqrt=None):
        Q = self.Q(k)
        ground_truth = self.ground_truth()
        if norm_matrix_sqrt is not None:
            Q = norm_matrix_sqrt @ Q
            ground_truth = norm_matrix_sqrt @ ground_truth
        # _, squared_l2_error, _, _ = lin.lstsq(Q, ground_truth, rcond=None)
        # return np.sqrt(squared_l2_error.item()))
        coeff = flamp.qr_solve(Q, ground_truth)
        # NOTE: when using norm_matrix_sqrt, it's already "baked" into `krylov_basis`
        # So at this point, we should just use the l2 norm:
        return norm(Q @ coeff - ground_truth)

    def spectrum_optimal_error(self, k, max_iter, tol):
        # Degree of polynomial must be strictly less than dimension of Krylov subspace used in Lanczos (so k - 1)
        return norm(self.b) * discrete_remez_error(
            degree=k-1, f_points=self.f(self.spectrum),
            points=self.spectrum, max_iter=max_iter, tol=tol
        )

    def pseudo_spectrum_optimal_error(self, k, max_iter, tol):
        augmented_spectrum = np.hstack((flamp.gmpy2.mpfr(0), self.spectrum))
        # Degree of polynomial must be strictly less than dimension of Krylov subspace used in Lanczos (so k - 1)
        return norm(self.b) * discrete_remez_error(
            degree=k-1, f_points=self.f(augmented_spectrum),
            points=augmented_spectrum, max_iter=max_iter, tol=tol
        )

    def fov_optimal_error_remez(self, k, max_iter, n_grid, tol):
        # Degree of polynomial must be strictly less than dimension of Krylov subspace used in Lanczos (so k - 1)
        return norm(self.b) * remez_error(
            degree=k-1, f=self.f, domain=(self.spectrum.min(), self.spectrum.max()),
            max_iter=max_iter, n_grid=n_grid, tol=tol
        )

    def fov_optimal_error_chebyshev_regression(self, k, num_points):
        spectrum_discritization = cheb_nodes(
            num_points, a=self.spectrum.min(), b=self.spectrum.max(), dtype=np.dtype('O'))
        f_spectrum_discritization = self.f(spectrum_discritization)
        # TODO: cache the creation of CV?
        CV = cheb_vandermonde(spectrum_discritization, k)
        cheb_coeffs = flamp.qr_solve(CV, f_spectrum_discritization)
        # cheb_coeffs, _, _, _ = lin.lstsq(CV[:, :k], f_spectrum_discritization, rcond=None)
        return norm(self.b) * norm(CV @ cheb_coeffs - f_spectrum_discritization, ord=np.inf)

    def adjuster(self, method_name, C_fun, k_fun, k, **kwargs):
        k = k_fun(self, k)
        if k < 1:
            return flamp.gmpy2.mpfr("inf")
        else:
            return C_fun(self, k) * getattr(self, method_name)(k=k, **kwargs)

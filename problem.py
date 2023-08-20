import numpy as np

import flamp

import matrix_functions as mf


class DiagonalFAProblem:
    def __init__(self, f, spectrum, b):
        self.f = f
        self.spectrum = spectrum
        self.b = b
        self.cached_decomp = None

    def ground_truth(self):
        return mf.diagonal_fa(self.f, self.spectrum, self.b)

    def A(self):
        return mf.DiagonalMatrix(self.spectrum)

    def sqrtA(self):
        return mf.DiagonalMatrix(flamp.sqrt(self.spectrum))

    def dim(self):
        return len(self.spectrum)

    def kappa(self):
        return np.abs(self.spectrum.max() / self.spectrum.min())

    def lanczos_decomp(self, k):
        if (self.cached_decomp is None) or (k > self.cached_decomp.Q.shape[1]):
            self.cached_decomp = mf.LanczosDecomposition.fit(self.A(), self.b, k, reorthogonalize=True)
        return self.cached_decomp.prefix(k)

    def Q(self, k):
        return self.lanczos_decomp(k).Q

    def lanczos_error(self, k, norm_matrix_sqrt=None):
        lanczos_estimate = self.lanczos_decomp(k).apply_function_to_start(self.f)
        error = lanczos_estimate - self.ground_truth()
        if norm_matrix_sqrt is None:
            return mf.norm(error)
        else:
            return mf.norm(norm_matrix_sqrt @ error)

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
        return mf.norm(Q @ coeff - ground_truth)

    def spectrum_optimal_error(self, k, max_iter, tol):
        # Degree of polynomial must be strictly less than dimension of Krylov subspace used in Lanczos (so k - 1)
        return mf.norm(self.b) * mf.discrete_remez_error(
            degree=k-1, f_points=self.f(self.spectrum),
            points=self.spectrum, max_iter=max_iter, tol=tol
        )

    def pseudo_spectrum_optimal_error(self, k, max_iter, tol):
        augmented_spectrum = np.concat((flamp.gmpy2.mpfr(0), self.spectrum))
        # Degree of polynomial must be strictly less than dimension of Krylov subspace used in Lanczos (so k - 1)
        return mf.norm(self.b) * mf.discrete_remez_error(
            degree=k-1, f_points=self.f(augmented_spectrum),
            points=augmented_spectrum, max_iter=max_iter, tol=tol
        )

    def fov_optimal_error_remez(self, k, max_iter, n_grid, tol):
        # Degree of polynomial must be strictly less than dimension of Krylov subspace used in Lanczos (so k - 1)
        return mf.norm(self.b) * mf.remez_error(
            degree=k-1, f=self.f, domain=(self.spectrum.min(), self.spectrum.max()),
            max_iter=max_iter, n_grid=n_grid, tol=tol
        )

    def fov_optimal_error_chebyshev_regression(self, k, num_points):
        spectrum_discritization = mf.cheb_nodes(
            num_points, a=self.spectrum.min(), b=self.spectrum.max(), dtype=np.dtype('O'))
        f_spectrum_discritization = self.f(spectrum_discritization)
        # TODO: cache the creation of CV?
        CV = mf.cheb_vandermonde(spectrum_discritization, k)
        cheb_coeffs = flamp.qr_solve(CV, f_spectrum_discritization)
        # cheb_coeffs, _, _, _ = lin.lstsq(CV[:, :k], f_spectrum_discritization, rcond=None)
        return mf.norm(self.b) * mf.norm(CV @ cheb_coeffs - f_spectrum_discritization, ord=np.inf)

    def adjuster(self, method_name, C_fun, k_fun, k, **kwargs):
        return C_fun(self, k) * getattr(self, method_name)(k=k_fun(self, k), **kwargs)


class InverseMonomial:
    def __init__(self, deg): self.deg = deg
    def __call__(self, x): return x**(-self.deg)
    def poles(self): return flamp.zeros(self.deg)
    def degree_denom(self): return self.deg


def fact1(problem, k, max_iter, n_grid, tol):
    return 2 * problem.fov_optimal_error_remez(k, max_iter=max_iter, n_grid=n_grid, tol=tol)


def thm1(problem, k):
    assert problem.spectrum.min() > 1
    assert np.all(problem.f.poles() <= 0)
    def C_fun(self, _): return self.f.degree_denom() * (self.kappa() ** self.f.degree_denom())
    def k_fun(self, k): return k - self.f.degree_denom() + 1
    return problem.adjuster("instance_optimal_error", C_fun, k_fun, k)


def thm2(problem, k, max_iter, tol):
    def C_fun(self, k): return 3 / np.sqrt(np.pi * k) * self.kappa()
    def k_fun(_, k): return k // 2
    if k >= 2:
        return problem.adjuster("spectrum_optimal_error", C_fun, k_fun, k, max_iter=max_iter, tol=tol)
    else:
        return np.inf


def thm3(problem, k, max_iter, tol):
    def C_fun(self, k): return 3 * self.kappa() ** 2 / (k ** (3/2))
    def k_fun(_, k): return k // 2 + 1
    if k >= 2:
        return problem.adjuster("pseudo_spectrum_optimal_error", C_fun, k_fun, k, max_iter=max_iter, tol=tol)
    else:
        return np.inf

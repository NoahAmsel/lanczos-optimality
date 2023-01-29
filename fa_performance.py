import flamp  # TODO: shouldn't automatically assume we're using flamp
import numpy as np
import pandas as pd
from tqdm import tqdm

import matrix_functions as mf


def lanczos_error_curve(f, A_decomp, ground_truth, ks):
    lanczos_errors = pd.Series(index=ks, dtype=np.float64)
    for k in tqdm(ks):
        lanczos_estimate = A_decomp.prefix(k).apply_function_to_start(f)
        lanczos_errors.loc[k] = mf.norm(lanczos_estimate - ground_truth)
    return lanczos_errors


def krylov_optimal_error_curve(krylov_basis, ground_truth, ks, norm_matrix_sqrt=None):
    if norm_matrix_sqrt is not None:
        krylov_basis = norm_matrix_sqrt @ krylov_basis
        ground_truth = norm_matrix_sqrt @ ground_truth

    krylov_errors = pd.Series(index=ks, dtype=np.float64)
    for k in tqdm(ks):
        # # _, squared_l2_error, _, _ = lin.lstsq(krylov_basis[:, :k], ground_truth, rcond=None)
        # # krylov_errors.loc[k] = np.sqrt(squared_l2_error.item()))
        _, residual = flamp.qr_solve(krylov_basis[:, :k], ground_truth, res=True)
        krylov_errors.loc[k] = mf.norm(residual)
    return krylov_errors


def chebyshev_interpolant_linf_error_curve(interval_lower, interval_upper, f, ks, num_points):
    spectrum_discritization = mf.cheb_nodes(num_points, a=interval_lower, b=interval_upper, dtype=np.dtype('O'))
    f_spectrum_discritization = f(spectrum_discritization)
    cheb_interpolant_errors = pd.Series(index=ks, dtype=np.float64)
    for k in tqdm(ks):
        # Degree of polynomial must be strictly less than dimension of Krylov subspace used in Lanczos (so k - 1)
        cheb_interpolant = mf.cheb_interpolation(k - 1, f, interval_lower, interval_upper, dtype=np.dtype('O'))
        cheb_interpolant_errors.loc[k] = mf.norm(
            cheb_interpolant(spectrum_discritization) - f_spectrum_discritization,
            ord=np.inf
        )
    return cheb_interpolant_errors


def chebyshev_regression_linf_error_curve(interval_lower, interval_upper, f, ks, num_points):
    spectrum_discritization = mf.cheb_nodes(num_points, a=interval_lower, b=interval_upper, dtype=np.dtype('O'))
    f_spectrum_discritization = f(spectrum_discritization)
    CV = mf.cheb_vandermonde(spectrum_discritization, max(ks))
    cheb_regression_errors = pd.Series(index=ks, dtype=np.float64)
    for k in tqdm(ks):
        cheb_coeffs = flamp.qr_solve(CV[:, :k], f_spectrum_discritization)
        # cheb_coeffs, _, _, _ = lin.lstsq(CV[:, :k], f_spectrum_discritization, rcond=None)
        cheb_regression_errors.loc[k] = mf.norm(CV[:, :k] @ cheb_coeffs - f_spectrum_discritization, ord=np.inf)

    return cheb_regression_errors


def fa_performance(f, a_diag, b, ks):
    ground_truth = mf.diagonal_fa(f, a_diag, b)
    A = mf.DiagonalMatrix(a_diag)
    A_decomp = mf.LanczosDecomposition.fit(A, b, max(ks), reorthogonalize=True)

    cols = dict()

    # Uniform bound, approximating the l-infinity approximant by...
    lambda_min = a_diag.min()
    lambda_max = a_diag.max()
    dim = len(a_diag)
    # ... Chebyshev interpolation
    unif_label = "$2||b|| \cdot \min_{\mathrm{deg}(p)<k} ||p - f||_{[\lambda_{\min}, \lambda_{\max}]}$"
    cols[unif_label] = 2 * mf.norm(b) * chebyshev_interpolant_linf_error_curve(
        lambda_min, lambda_max, f, ks, 10 * dim)
    # # Chebyshev regression
    # cols["Chebyshev regression $* 2||x||$"] = 2 * mf.norm(x) * chebyshev_regression_linf_error_curve(
    #     lambda_min, lambda_max, f, ks, 10 * dim)

    # Lanczos-FA
    cols["$||\mathrm{lan}_k - f(A)b||_2$"] = lanczos_error_curve(f, A_decomp, ground_truth, ks)

    # Optimal approximation to ground truth in Krylov subspace...
    # ...with respect to either Euclidean norm
    cols["$||\mathrm{opt}_k(I) - f(A)b||_2$"] = krylov_optimal_error_curve(A_decomp.Q, ground_truth, ks)
    # # ...with respect to A norm
    # sqrtA = mf.DiagonalMatrix(flamp.sqrt(a_diag))
    # cols["Krlov subspace (A-norm)"] = krylov_optimal_error_curve(A_decomp.Q, ground_truth, ks, sqrtA)

    results = pd.concat(cols, axis=1)
    assert (results != flamp.gmpy2.mpfr('nan')).all().all()
    assert (~pd.isna(results)).all().all()
    return results

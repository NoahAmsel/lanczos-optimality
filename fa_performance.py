import flamp  # TODO: shouldn't automatically assume we're using flamp
import numpy as np
import pandas as pd

import matrix_functions as mf


def lanczos_error_curve(f, A_decomp, ground_truth, ks, norm_matrix_sqrt=None):
    lanczos_errors = pd.Series(index=ks, dtype=np.dtype('O'))
    for k in ks:
        lanczos_estimate = A_decomp.prefix(k).apply_function_to_start(f)
        error = lanczos_estimate - ground_truth
        lanczos_errors.loc[k] = mf.norm(error) if norm_matrix_sqrt is None else mf.norm(norm_matrix_sqrt @ error)
    return lanczos_errors


def krylov_optimal_error_curve(krylov_basis, ground_truth, ks, norm_matrix_sqrt=None):
    if norm_matrix_sqrt is not None:
        krylov_basis = norm_matrix_sqrt @ krylov_basis
        ground_truth = norm_matrix_sqrt @ ground_truth

    krylov_errors = pd.Series(index=ks, dtype=np.dtype('O'))
    for k in ks:
        # # _, squared_l2_error, _, _ = lin.lstsq(krylov_basis[:, :k], ground_truth, rcond=None)
        # # krylov_errors.loc[k] = np.sqrt(squared_l2_error.item()))
        coeff = flamp.qr_solve(krylov_basis[:, :k], ground_truth)
        residual = krylov_basis[:, :k] @ coeff - ground_truth
        krylov_errors.loc[k] = mf.norm(residual) if norm_matrix_sqrt is None else mf.norm(norm_matrix_sqrt @ residual)
    return krylov_errors


def chebyshev_interpolant_linf_error_curve(interval_lower, interval_upper, f, ks, num_points):
    spectrum_discritization = mf.cheb_nodes(num_points, a=interval_lower, b=interval_upper, dtype=np.dtype('O'))
    f_spectrum_discritization = f(spectrum_discritization)
    cheb_interpolant_errors = pd.Series(index=ks, dtype=np.dtype('O'))
    for k in ks:
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
    cheb_regression_errors = pd.Series(index=ks, dtype=np.dtype('O'))
    for k in ks:
        cheb_coeffs = flamp.qr_solve(CV[:, :k], f_spectrum_discritization)
        # cheb_coeffs, _, _, _ = lin.lstsq(CV[:, :k], f_spectrum_discritization, rcond=None)
        cheb_regression_errors.loc[k] = mf.norm(CV[:, :k] @ cheb_coeffs - f_spectrum_discritization, ord=np.inf)

    return cheb_regression_errors


def our_bound_curve(krylov_basis, ground_truth, ks, denom_degree, kappa):
    our_bound = pd.Series(data=np.inf, index=ks, dtype=np.dtype('O'))
    # Bound doesn't exist unless k - denom_degree + 1 > 0
    filtered_ks = list(filter(lambda k: k - denom_degree + 1 > 0, ks))
    krylov_optimal = krylov_optimal_error_curve(
        krylov_basis, ground_truth, np.array(filtered_ks) - denom_degree + 1, norm_matrix_sqrt=None)
    for k in filtered_ks:
        our_bound.loc[k] = denom_degree * (kappa ** denom_degree) * krylov_optimal.loc[k - denom_degree + 1]
    return our_bound


def fa_performance(f, a_diag, b, ks, relative_error=True,
                   uniform_bound_interpolation=True,
                   uniform_bound_regression=False,
                   lanczos=True,
                   lanczos_Anorm=False,
                   krylov_optimal=True,
                   krylov_optimal_Anorm=False,
                   our_bound=True):

    ground_truth = mf.diagonal_fa(f, a_diag, b)
    A = mf.DiagonalMatrix(a_diag)
    A_decomp = mf.LanczosDecomposition.fit(A, b, max(ks), reorthogonalize=True)
    sqrtA = mf.DiagonalMatrix(flamp.sqrt(a_diag))

    cols = dict()

    # Uniform bound, approximating the l-infinity approximant by...
    lambda_min = a_diag.min()
    lambda_max = a_diag.max()
    dim = len(a_diag)

    if uniform_bound_interpolation:
        # ... Chebyshev interpolation
        unif_label = r"$2||b|| \cdot \min_{\mathrm{deg}(p)<k} ||p - f||_{[\lambda_{\min}, \lambda_{\max}]}$"
        cols[unif_label] = 2 * mf.norm(b) * chebyshev_interpolant_linf_error_curve(
            lambda_min, lambda_max, f, ks, 10 * dim)
    if uniform_bound_regression:
        # Chebyshev regression
        cols[r"Chebyshev regression $\cdot 2||b||$"] = 2 * mf.norm(b) * chebyshev_regression_linf_error_curve(
            lambda_min, lambda_max, f, ks, 10 * dim)

    if lanczos:
        # Lanczos-FA
        cols[r"$||\mathrm{lan}_k - f(A)b||_2$"] = lanczos_error_curve(f, A_decomp, ground_truth, ks)
    if lanczos_Anorm:
        # Lanczos-FA with error measured in A norm
        cols[r"$||\mathrm{lan}_k - f(A)b||_A$"] = lanczos_error_curve(f, A_decomp, ground_truth, ks, sqrtA)

    # Optimal approximation to ground truth in Krylov subspace...
    if krylov_optimal:
        # ...with respect to Euclidean norm
        cols[r"$||\mathrm{opt}_k(I) - f(A)b||_2$"] = krylov_optimal_error_curve(A_decomp.Q, ground_truth, ks)
    if krylov_optimal_Anorm:
        # ...with respect to A norm
        cols[r"$||\mathrm{opt}_k(A) - f(A)b||_A$"] = krylov_optimal_error_curve(A_decomp.Q, ground_truth, ks, sqrtA)

    if our_bound:
        assert hasattr(f, "degree")
        cols["Our Bound"] = our_bound_curve(A_decomp.Q, ground_truth, ks, f.degree[1], lambda_max / lambda_min)

    results = pd.concat(cols, axis=1)
    # notice that it's relative to the *Euclidean* norm of the ground truth
    if relative_error:
        results /= mf.norm(ground_truth)

    assert (results != flamp.gmpy2.mpfr('nan')).all().all()
    assert (~pd.isna(results)).all().all()

    return results

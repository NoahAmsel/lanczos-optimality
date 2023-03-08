import flamp  # TODO: shouldn't automatically assume we're using flamp
import numpy as np
import pandas as pd

import baryrat

import matrix_functions as mf
from remez import remez_flamp

from tqdm import tqdm


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
        error = krylov_basis[:, :k] @ coeff - ground_truth
        # NOTE: when using norm_matrix_sqrt, it's already "baked" into `krylov_basis`
        # So at this point, we should just use the l2 norm:
        krylov_errors.loc[k] = mf.norm(error)
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


def remez_error_curve(interval_lower, interval_upper, f, ks, max_iter=100, n_grid=2000, tol=1e-10):
    remez_errors = pd.Series(flamp.gmpy2.mpfr('inf') * len(ks), index=ks, dtype=np.dtype('O'))
    for k in ks:
        try:
            _, error_upper_bound, _ = remez_flamp(f, k, domain=[interval_lower, interval_upper], max_iter=max_iter, n_grid=n_grid, tol=tol)
            remez_errors.loc[k] = error_upper_bound
        except Exception:
            # Yeah Remez gets stuck sometimes
            pass
    return remez_errors


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
                   remez_uniform=False,
                   lanczos=True,
                   lanczos_Anorm=False,
                   lanczos_alt_norms=None,
                   krylov_optimal=True,
                   krylov_optimal_Anorm=False,
                   krylov_optimal_alt_norms=None,
                   our_bound=True):
    # TODO: add a "tqdm" option that, instead of passing ks below, passes tqdm(ks)

    ground_truth = mf.diagonal_fa(f, a_diag, b)
    A = mf.DiagonalMatrix(a_diag)
    A_decomp = mf.LanczosDecomposition.fit(A, b, max(ks), reorthogonalize=True)
    sqrtA = mf.DiagonalMatrix(flamp.sqrt(a_diag))

    cols = dict()

    # Uniform bound, approximating the l-infinity approximant by...
    lambda_min = a_diag.min()
    lambda_max = a_diag.max()
    dim = len(a_diag)

    if our_bound:
        assert hasattr(f, "degree")
        cols["Our Bound"] = our_bound_curve(A_decomp.Q, ground_truth, ks, f.degree[1], lambda_max / lambda_min)

    if uniform_bound_interpolation:
        # ... Chebyshev interpolation
        unif_label = r"$2||b||_2 \cdot \min_{\mathrm{deg}(p)<k} \|p(x) - f(x)\|_{\infty, [\lambda_{\min}, \lambda_{\max}]}$"
        cols[unif_label] = 2 * mf.norm(b) * chebyshev_interpolant_linf_error_curve(
            lambda_min, lambda_max, f, ks, 10 * dim)
        # the error of the best approximation is monotonically decreasing with degree
        # but since this is only an approximation, we might have to enforce this property
        cols[unif_label] = np.minimum.accumulate(cols[unif_label])
    if uniform_bound_regression:
        # Chebyshev regression
        cols[r"Chebyshev regression $\cdot 2||b||$"] = 2 * mf.norm(b) * chebyshev_regression_linf_error_curve(
            lambda_min, lambda_max, f, ks, 10 * dim)
        # the error of the best approximation is monotonically decreasing with degree
        # but since this is only an approximation, we might have to enforce this property
        cols[r"Chebyshev regression $\cdot 2||b||$"] = np.minimum.accumulate(cols[r"Chebyshev regression $\cdot 2||b||$"])

    if remez_uniform:
        cols["Remez Uniform"] = 2 * mf.norm(b) * remez_error_curve(lambda_min, lambda_max, f, ks, max_iter=100, n_grid=2000, tol=1e-10)
        cols["Remez Uniform"] = np.minimum.accumulate(cols["Remez Uniform"])

    if lanczos:
        # Lanczos-FA
        cols[r"$||\mathrm{lan}_k - f(A)b||_2$"] = lanczos_error_curve(f, A_decomp, ground_truth, ks)
    if lanczos_Anorm:
        # Lanczos-FA with error measured in A norm
        cols[r"$||\mathrm{lan}_k - f(A)b||_A$"] = lanczos_error_curve(f, A_decomp, ground_truth, ks, sqrtA)
    if lanczos_alt_norms is not None:
        for norm_name, norm_matrix_sqrt in lanczos_alt_norms.items():
            cols[rf"$||\mathrm{{lan}}_k - f(A)b||_{norm_name}$"] = lanczos_error_curve(f, A_decomp, ground_truth, ks, norm_matrix_sqrt)

    # Optimal approximation to ground truth in Krylov subspace...
    if krylov_optimal:
        # ...with respect to Euclidean norm
        cols[r"$||\mathrm{opt}_k(I) - f(A)b||_2$"] = krylov_optimal_error_curve(A_decomp.Q, ground_truth, ks)
    if krylov_optimal_Anorm:
        # ...with respect to A norm
        cols[r"$||\mathrm{opt}_k(A) - f(A)b||_A$"] = krylov_optimal_error_curve(A_decomp.Q, ground_truth, ks, sqrtA)
    if krylov_optimal_alt_norms is not None:
        for norm_name, norm_matrix_sqrt in krylov_optimal_alt_norms.items():
            cols[rf"$||\mathrm{{opt}}_k(A) - f(A)b||_{norm_name}$"] = krylov_optimal_error_curve(A_decomp.Q, ground_truth, ks, norm_matrix_sqrt)

    results = pd.concat(cols, axis=1)
    # notice that it's relative to the *Euclidean* norm of the ground truth
    if relative_error:
        results /= mf.norm(ground_truth)

    assert (results != flamp.gmpy2.mpfr('nan')).all().all()
    assert (~pd.isna(results)).all().all()

    return results


def fun_vs_rationals(f, a_diag, b, ks, degrees, approximator=None, fname="f", relative_error=True):
    if approximator is None:
        def approximator(degree):
            return baryrat.brasil(f, (a_diag.min(), a_diag.max()), degree, tol=0.0001, info=False)

    ground_truth = mf.diagonal_fa(f, a_diag, b)
    A = mf.DiagonalMatrix(a_diag)
    A_decomp = mf.LanczosDecomposition.fit(A, b, max(ks), reorthogonalize=True)

    cols = dict()
    # deg2unif_errors = dict()

    cols[fname] = lanczos_error_curve(f, A_decomp, ground_truth, ks)
    # cols["Krylov optimal"] = krylov_optimal_error_curve(A_decomp.Q, ground_truth, ks)

    for degree in tqdm(degrees):
        approximant = approximator(degree)
        # approximant, info = baryrat.brasil(f, (a_diag.min(), a_diag.max()), degree, tol=0.0001, info=True)
        # assert info.converged
        cols[f"deg={degree}"] = lanczos_error_curve(approximant, A_decomp, ground_truth, ks)
        # deg2unif_errors[degree] = info.error

    results = pd.concat(cols, axis=1)
    # uniform_errors = pd.Series(deg2unif_errors)
    # notice that it's relative to the *Euclidean* norm of the ground truth
    if relative_error:
        results /= mf.norm(ground_truth)
        # uniform_errors /= mf.norm(ground_truth)

    assert (results != flamp.gmpy2.mpfr('nan')).all().all()
    assert (~pd.isna(results)).all().all()

    # return results, uniform_errors
    return results

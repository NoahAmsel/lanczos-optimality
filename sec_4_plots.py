import matplotlib.pyplot as plt
import seaborn as sns

from problem import *


def plot_convergence_curves(error_df, relative_error=True, **kwargs):
    if relative_error:
        error_label = "Relative Error"
    else:
        error_label = "Error"

    k_label = "Number of iterations ($k$)"
    error_df_long = error_df.reset_index(names=k_label).melt(id_vars=[k_label], value_name=error_label, var_name="Line")

    return sns.lineplot(
        x=k_label,
        y=error_label,
        data=error_df_long,
        hue="Line",
        style="Line",
        size="Line" if ("sizes" in kwargs) else None,
        **kwargs
    ).set(
        title=kwargs.get("title", None),
        xlabel=k_label,
        ylabel=error_label,
        yscale='log'
    )


if __name__ == "__main__":
    import pandas as pd

    flamp.set_dps(300)  # compute with this many decimal digits precision
    print(f"Using {flamp.get_dps()} digits of precision")

    dim = 100
    kappa = flamp.gmpy2.mpfr(100.)
    lambda_min = flamp.gmpy2.mpfr(1.)
    # a_diag = flamp.linspace(lambda_min, kappa*lambda_min, dim)
    # a_diag = mf.geometric_spectrum(dim, kappa, rho=1e-3, lambda_1=lambda_min)
    a_diag = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=5, lambda_1=lambda_min)
    # b = flamp.to_mp(mf.geometric_spectrum(dim, 1e4, 1000))
    # b = flamp.to_mp(mf.geometric_spectrum(dim, 1e4, 1000))
    b = flamp.ones(dim)
    def f(x): return 1 / flamp.sqrt(x)
    p = DiagonalFAProblem(f, a_diag, b)

    # ks = list(range(1, 61))
    ks = list(range(1, 61, 10))
    # for caching purposes
    p.lanczos_decomp(max(ks))

    relative_error_df = pd.DataFrame(index=ks, data={
        "FOV Optimal": [fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14) for k in ks],
        "Thm 2": [thm2(p, k, max_iter=100, tol=1e-14) for k in ks],
        "Lanczos-FA": [p.lanczos_error(k) for k in ks],
        "Instance Optimal": [p.instance_optimal_error(k) for k in ks]
    }) / mf.norm(p.ground_truth())
    plot_convergence_curves(relative_error_df, relative_error=True)
    plt.show()

    # relative_error_df = pd.DataFrame(index=ks, data={
    #     "FOV Optimal": [fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14) for k in ks],
    #     "Spectrum Optimal": [p.spectrum_optimal_error(k, max_iter=100, tol=1e-14) for k in ks],
    #     "Lanczos-FA": [p.lanczos_error(k) for k in ks],
    #     "Instance Optimal": [p.instance_optimal_error(k) for k in ks]
    # }) / mf.norm(p.ground_truth())

import flamp
import matplotlib.pyplot as plt
import seaborn as sns

import matrix_functions as mf
from problem import *


def plot_convergence_curves(error_df, relative_error=True, **kwargs):
    if relative_error:
        error_label = "Relative Error"
    else:
        error_label = "Error"

    k_label = "Number of iterations ($k$)"
    error_df_long = error_df.reset_index(names=k_label).melt(id_vars=[k_label], value_name=error_label, var_name="Line")

    if "title" in kwargs:
        title = kwargs.pop("title")
    else:
        title = None

    return sns.lineplot(
        x=k_label,
        y=error_label,
        data=error_df_long,
        hue="Line",
        style="Line",
        size="Line" if ("sizes" in kwargs) else None,
        **kwargs
    ).set(
        title=title,
        xlabel=k_label,
        ylabel=error_label,
        yscale='log'
    )


def inv_sqrt_data():
    dim = 100
    kappa = flamp.gmpy2.mpfr(100.)
    lambda_min = flamp.gmpy2.mpfr(1.)
    a_diag = mf.geometric_spectrum(dim, kappa, rho=1e-3, lambda_1=lambda_min)
    b = flamp.ones(dim)
    def f(x): return 1 / flamp.sqrt(x)
    ks = list(range(1, 61))
    p = DiagonalFAProblem(f, a_diag, b, cache_k=max(ks))

    relative_error_df = pd.DataFrame(index=ks, data={
        "FOV Optimal": [fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14) for k in tqdm(ks)],
        "Theorem 2": [thm2(p, k, max_iter=100, tol=1e-14) for k in tqdm(ks)],
        "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
        "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)]
    }) / mf.norm(p.ground_truth())
    return relative_error_df


def sqrt_data():
    dim = 100
    kappa = flamp.gmpy2.mpfr(100.)
    lambda_min = flamp.gmpy2.mpfr(1.)
    a_diag = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=10, lambda_1=lambda_min)
    b = flamp.ones(dim)
    f = flamp.sqrt
    ks = list(range(1, 61))
    p = DiagonalFAProblem(f, a_diag, b, cache_k=max(ks))

    # for caching purposes
    p.lanczos_decomp(max(ks))

    relative_error_df = pd.DataFrame(index=ks, data={
        "FOV Optimal": [fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14) for k in ks],
        "Theorem 3": [thm3(p, k, max_iter=100, tol=1e-14) for k in ks],
        "Lanczos-FA": [p.lanczos_error(k) for k in ks],
        "Instance Optimal": [p.instance_optimal_error(k) for k in ks]
    }) / mf.norm(p.ground_truth())
    return relative_error_df


if __name__ == "__main__":
    import pandas as pd

    flamp.set_dps(300)  # compute with this many decimal digits precision
    print(f"Using {flamp.get_dps()} digits of precision")

    inv_sqrt_relative_error_df = inv_sqrt_data()
    inv_sqrt_relative_error_df.to_csv("output/paper_data/inv_sqrt_data.csv", index=False)
    inv_sqrt_relative_error_df = pd.read_csv("output/paper_data/inv_sqrt_data.csv")

    sqrt_relative_error_df = sqrt_data()
    sqrt_relative_error_df.to_csv("output/paper_data/sqrt_data.csv", index=False)
    sqrt_relative_error_df = pd.read_csv("output/paper_data/sqrt_data.csv")

    # sns.set(font_scale=2)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    style_df = pd.DataFrame({
        "FOV Optimal": [(3, 1), 1.5, sns.color_palette("rocket", 4)[1]],
        "Theorem 2": [(3, 1, 1, 1), 1.5, sns.color_palette("rocket", 4)[3]],
        "Theorem 3": [(3, 1, 1, 1), 1.5, sns.color_palette("rocket", 4)[3]],
        "Lanczos-FA": [(1, 1), 3, sns.color_palette("rocket", 4)[2]],
        "Instance Optimal": [(1, 0), 1, sns.color_palette("rocket", 4)[0]],
    }, index=["dashes", "sizes", "palette"])

    fig, axs = plt.subplots(
        1, 2, figsize=(8, 4), squeeze=True
    )
    plot_convergence_curves(inv_sqrt_relative_error_df, relative_error=True,
                            ax=axs[0], title=r"$\mathbf A^{-1/2}\mathbf b$", **style_df.transpose().to_dict())
    plot_convergence_curves(sqrt_relative_error_df, relative_error=True,
                            ax=axs[1], title=r"$\mathbf A^{1/2}\mathbf b$", **style_df.transpose().to_dict())
    for ax in axs:
        ax.set(xlabel=None)
        ax.legend(title='')  # loc='upper center', ncol=4, bbox_to_anchor=(1, 0)
    for ax in axs[1:]:
        ax.set(ylabel=None)
        # axs[1].legend([], [], frameon=False)

    fig.supxlabel("Number of iterations ($k$)")
    fig.tight_layout()
    fig.savefig("output/paper_plots/sec4.svg")

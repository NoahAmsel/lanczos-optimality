import baryrat
import flamp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
from tqdm import tqdm

import matrix_functions as mf


class InverseMonomial:
    def __init__(self, deg): self.deg = deg
    def __call__(self, x): return x**(-self.deg)
    def poles(self): return flamp.zeros(self.deg)
    def degree_denom(self): return self.deg


def fact1(problem, k, max_iter, n_grid, tol):
    return 2 * problem.fov_optimal_error_remez(k, max_iter=max_iter, n_grid=n_grid, tol=tol)


def thm1(problem, k):
    lmin = problem.spectrum.min()
    lmax = problem.spectrum.max()
    poles = problem.f.poles()
    if all(abs(pole.imag) < 1e-16 for pole in poles):
        poles = flamp.to_mp([pole.real for pole in poles])
    low_poles = poles[poles < lmin]
    high_poles = poles[lmax < poles]
    assert len(low_poles) + len(high_poles) == len(poles), "Poles in the range of the eigenvalues are not allowed!"
    kappa_part = np.prod((lmax - low_poles) / (lmin - low_poles)) * np.prod((high_poles - lmin) / (high_poles - lmax))
    def C_fun(self, _): return self.f.degree_denom() * kappa_part
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
    p = mf.DiagonalFAProblem(f, a_diag, b, cache_k=max(ks))

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
    p = mf.DiagonalFAProblem(f, a_diag, b, cache_k=max(ks))

    relative_error_df = pd.DataFrame(index=ks, data={
        "FOV Optimal": [fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14) for k in tqdm(ks)],
        "Theorem 3": [thm3(p, k, max_iter=100, tol=1e-14) for k in tqdm(ks)],
        "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
        "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)]
    }) / mf.norm(p.ground_truth())
    return relative_error_df


def sqrt_inv_sqrt_plot():
    inv_sqrt_relative_error_df = inv_sqrt_data()
    inv_sqrt_relative_error_df.to_csv("output/paper_data/inv_sqrt_data.csv", index=False)
    inv_sqrt_relative_error_df = pd.read_csv("output/paper_data/inv_sqrt_data.csv")

    sqrt_relative_error_df = sqrt_data()
    sqrt_relative_error_df.to_csv("output/paper_data/sqrt_data.csv", index=False)
    sqrt_relative_error_df = pd.read_csv("output/paper_data/sqrt_data.csv")

    fig, axs = plt.subplots(
        1, 2, figsize=(8, 4), squeeze=True
    )
    plot_convergence_curves(inv_sqrt_relative_error_df, relative_error=True,
                            ax=axs[0], title=r"$\mathbf A^{-1/2}\mathbf b$", **master_style_df.transpose().to_dict())
    plot_convergence_curves(sqrt_relative_error_df, relative_error=True,
                            ax=axs[1], title=r"$\mathbf A^{1/2}\mathbf b$", **master_style_df.transpose().to_dict())
    for ax in axs:
        ax.set(xlabel=None)
        ax.legend(title='')  # loc='upper center', ncol=4, bbox_to_anchor=(1, 0)
    for ax in axs[1:]:
        ax.set(ylabel=None)
        # axs[1].legend([], [], frameon=False)

    fig.supxlabel("Number of iterations ($k$)")
    fig.tight_layout()
    fig.savefig("output/paper_plots/sec4.svg")
    return fig


def general_performance_data():
    dim = 100
    kappa = flamp.gmpy2.mpfr(100.)
    lambda_min = flamp.gmpy2.mpfr(1.)
    a_diag_unif = flamp.linspace(lambda_min, kappa*lambda_min, dim)
    a_diag_geom = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
    a_diag_two_cluster = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=10, lambda_1=lambda_min)
    geom_b = flamp.to_mp(mf.geometric_spectrum(dim, 1e4, 1000))
    ks = list(range(1, 61))
    problems = {
        r"$\mathbf A^{-2}\mathbf b$": mf.DiagonalFAProblem(InverseMonomial(2), a_diag_unif, geom_b, cache_k=max(ks)),
        r"$\exp(\mathbf A)\mathbf b$": mf.DiagonalFAProblem(flamp.exp, a_diag_geom, geom_b, cache_k=max(ks)),
        r"$\log(\mathbf A)\mathbf b$": mf.DiagonalFAProblem(flamp.log, a_diag_two_cluster, geom_b, cache_k=max(ks)),
    }
    relative_error_dfs = {
        label: pd.DataFrame(index=ks, data={
            "FOV Optimal": [fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14) for k in tqdm(ks)],
            "Spectrum Optimal": [p.spectrum_optimal_error(k, max_iter=100, tol=1e-14) for k in tqdm(ks)],
            "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
            "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)]
        }) / mf.norm(p.ground_truth()) for label, p in problems.items()
    }
    return relative_error_dfs


def general_performance_plot():
    data = general_performance_data()
    with open("output/paper_data/general_performance_data.pkl", "wb") as f:
        pkl.dump(data, f)
    with open("output/paper_data/general_performance_data.pkl", "rb") as f:
        data = pkl.load(f)
    fig_scale = 1.5  # default. shouldn't matter when using svg
    row_heights = [1, 0.25]
    fig_width = fig_scale * len(data) * .9 + 4
    fig_height = sum(row_heights) * fig_scale * 2.5
    fig, axs = plt.subplots(
        2, len(data),
        sharex=True,
        height_ratios=row_heights,
        figsize=(fig_width, fig_height),
    )
    for i, (label, relative_error_df) in enumerate(data.items()):
        plot_convergence_curves(
            relative_error_df,
            relative_error=True,
            ax=axs[0, i],
            title=label,
            **master_style_df.transpose().to_dict()
        )

        optimality_ratios = relative_error_df["Lanczos-FA"] / relative_error_df["Instance Optimal"]

        sns.lineplot(data=optimality_ratios, ax=axs[1, i], lw=1.5).set(
            ylabel="Optimality Ratio" if (i == 0) else None
        )
        axs[0, i].set(xlabel=None)
        axs[0, i].legend(title='')  # loc='upper center', ncol=4, bbox_to_anchor=(1, 0)
        if i > 0:
            axs[0, i].set(ylabel=None)
            axs[0, i].legend([], [], frameon=False)

    fig.supxlabel("Number of iterations ($k$)")
    fig.tight_layout()
    fig.savefig("output/paper_plots/general_performance.svg")
    return fig


def our_bound_data():
    dim = 100
    kappa = flamp.gmpy2.mpfr(100.)
    lambda_min = flamp.gmpy2.mpfr(1.)
    a_diag_unif = flamp.linspace(lambda_min, kappa*lambda_min, dim)
    a_diag_geom = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
    a_diag_two_cluster = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=10, lambda_1=lambda_min)
    b = flamp.ones(dim)
    ks = list(range(1, 61))
    problems = {
        r"$\mathbf A^{-2}\mathbf b$": mf.DiagonalFAProblem(InverseMonomial(2), a_diag_unif, b, cache_k=max(ks)),
        r"$r(\mathbf A)\mathbf b \approx \mathbf A^{.9} \mathbf b$ (deg=5)": mf.DiagonalFAProblem(baryrat.brasil(lambda x: x**(.9), (flamp.gmpy2.mpfr(1), flamp.gmpy2.mpfr(kappa)), 5, info=False), a_diag_geom, b, cache_k=max(ks)),
        r"$r(\mathbf A)\mathbf b \approx \log(\mathbf A)\mathbf b$ (deg=10)": mf.DiagonalFAProblem(baryrat.brasil(flamp.log, (flamp.gmpy2.mpfr(1), flamp.gmpy2.mpfr(kappa)), 10, info=False), a_diag_two_cluster, b, cache_k=max(ks)),
    }
    relative_error_dfs = {
        label: pd.DataFrame(index=ks, data={
            "FOV Optimal": [fact1(p, k, max_iter=100, n_grid=1000, tol=1e-14) for k in tqdm(ks)],
            "Theorem 1": [thm1(p, k) for k in tqdm(ks)],
            "Lanczos-FA": [p.lanczos_error(k) for k in tqdm(ks)],
            "Instance Optimal": [p.instance_optimal_error(k) for k in tqdm(ks)]
        }) / mf.norm(p.ground_truth()) for label, p in problems.items()
    }
    return relative_error_dfs


def our_bound_plot():
    data = our_bound_data()
    with open("output/paper_data/our_bound_data.pkl", "wb") as f:
        pkl.dump(data, f)
    with open("output/paper_data/our_bound_data.pkl", "rb") as f:
        data = pkl.load(f)
    fig_scale = 1.5  # default. shouldn't matter when using svg
    fig_width = fig_scale * len(data) * .9 + 4
    fig_height = fig_scale * 2.5
    fig, axs = plt.subplots(
        1, len(data),
        sharex=True,
        figsize=(fig_width, fig_height),
        squeeze=True
    )
    for i, (label, relative_error_df) in enumerate(data.items()):
        plot_convergence_curves(
            relative_error_df,
            relative_error=True,
            ax=axs[i],
            title=label,
            **master_style_df.transpose().to_dict()
        )

        axs[i].set(xlabel=None)
        axs[i].legend(title='')  # loc='upper center', ncol=4, bbox_to_anchor=(1, 0)
        if i > 0:
            axs[i].set(ylabel=None)
            axs[i].legend([], [], frameon=False)

    fig.supxlabel("Number of iterations ($k$)")
    fig.tight_layout()
    fig.savefig("output/paper_plots/our_bound.svg")
    return fig


if __name__ == "__main__":
    flamp.set_dps(300)  # compute with this many decimal digits precision
    print(f"Using {flamp.get_dps()} digits of precision")

    # sns.set(font_scale=2)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    our_bound = [(3, 1, 1, 1), 1.5, sns.color_palette("rocket", 4)[3]]
    master_style_df = pd.DataFrame({
        "FOV Optimal": [(3, 1), 1.5, sns.color_palette("rocket", 4)[1]],
        "Spectrum Optimal": [(2, 1, 1, 1, 1, 1), 1.5, sns.color_palette("tab10")[-1]],
        "Theorem 1": our_bound,
        "Theorem 2": our_bound,
        "Theorem 3": our_bound,
        "Lanczos-FA": [(1, 1), 3, sns.color_palette("rocket", 4)[2]],
        "Instance Optimal": [(1, 0), 1, sns.color_palette("rocket", 4)[0]],
    }, index=["dashes", "sizes", "palette"])

    # sqrt_inv_sqrt_plot()
    # general_performance_plot()
    our_bound_plot()

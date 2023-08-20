import baryrat
import flamp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import matrix_functions as mf
from fa_performance import fa_performance, fun_vs_rationals
from zolotarev import tylers_sqrt, exp_pade0_55


def plot_convergence_curves(results, error_label, k_label, title=None, dashes=True, sizes=None, markers=None, ax=None):
    str_error_label = "error" if error_label is None else error_label
    str_k_label = "k" if k_label is None else k_label

    results_long = results.reset_index(names=str_k_label).melt(
        id_vars=[str_k_label], value_name=str_error_label, var_name="Function")

    sns.lineplot(
        x=str_k_label,
        y=str_error_label,
        hue="Function",
        style="Function",
        size="Function" if (sizes is not None) else None,
        dashes=dashes,
        sizes=sizes,
        # markers=markers,
        data=results_long,
        markeredgecolor="None",
        markersize=10,
        lw=1.5,
        ax=ax,
    ).set(
        title=title,
        xlabel=k_label,
        ylabel=error_label,
        yscale='log'
    )

# superplot_counter = 0


def convergence_superplot(a_diag, b, ks, functions_dict, relative_error, plot_our_bound=True, plot_optimality_ratios=True, differentiate_sizes=True, plot_unif=True, plot_spectrum_optimal=False):
    # global superplot_counter
    # superplot_counter += 1
    fig_scale = 1.5  # default. shouldn't matter when using svg
    row_heights = [1] + ([0.25] if plot_optimality_ratios else [])
    fig_width = fig_scale * len(functions_dict) * .9 + 4
    fig_height = sum(row_heights) * fig_scale * 2.5
    fig, axs = plt.subplots(
        len(row_heights), len(functions_dict),
        sharex=True,
        height_ratios=row_heights,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    if relative_error:
        error_label = "Relative Error"
    else:
        error_label = "Error"
    # TODO: is this label actually right?
    k_label = "Number of iterations ($k$)"
    # k_label = "Number of matrix-vector products ($k$)"

    krylov_optimal_label = r"$||\mathrm{opt}_k(I) - f(A)b||_2$"
    lanczos_label = r"$||\mathrm{lan}_k - f(A)b||_2$"
    unif_label = r"$2||b||_2 \cdot \min_{\mathrm{deg}(p)<k} \|p(x) - f(x)\|_{\infty, [\lambda_{\min}, \lambda_{\max}]}$"
    cheb_unif_label = r"Chebyshev regression $\cdot 2||b||$"
    spec_opt_label = r"$||b||_2 \cdot \min_{\mathrm{deg}(p)<k} \max_{x \in \Lambda} \|p(x) - f(x)\|$"
    our_label = "Our Bound"

    for fun_ix, (fun_label, fun) in enumerate(tqdm(functions_dict.items())):
        results = fa_performance(fun, a_diag[fun_ix], b, ks,
                                 relative_error=relative_error,
                                 remez_uniform=plot_unif, uniform_bound_regression=False,
                                 spectrum_optimal=plot_spectrum_optimal,
                                 our_bound=plot_our_bound)
        # results.to_csv(f"output/paper_data/superplot{superplot_counter}_{fun_label}.csv")

        results.rename(columns={
            krylov_optimal_label: "Instance Optimal",
            lanczos_label: "Lanczos-FA",
            unif_label: "FOV Optimal",
            # cheb_unif_label: "FOV Optimal",
            spec_opt_label: "Spectrum Optimal",
            our_label: "Our Bound"
            }, inplace=True)
        results = results[results.columns[::-1]]

        if (not plot_unif) and ("FOV Optimal" in list(results)):
            results.drop("FOV Optimal", axis=1, inplace=True)

        plot_convergence_curves(
            results,
            error_label=error_label if (fun_ix == 0) else None,
            k_label=None,
            title=fun_label,
            dashes={
                "Lanczos-FA": (1, 1),
                "Instance Optimal": (1, 0),
                "FOV Optimal": (3, 1),
                cheb_unif_label: (3, 1),
                "Spectrum Optimal": (2, 1, 1, 1, 1, 1),
                "Our Bound": (3, 1, 1, 1),
            },
            sizes={"Lanczos-FA": 3, "Instance Optimal": 1, "FOV Optimal": 1.5, cheb_unif_label: 1.5, "Spectrum Optimal": 1.5, "Our Bound": 1.5} if differentiate_sizes else None,
            ax=axs[0, fun_ix]
        )

        if plot_optimality_ratios:
            optimality_ratios = results["Lanczos-FA"] / results["Instance Optimal"]
            sns.lineplot(data=optimality_ratios, ax=axs[1, fun_ix], lw=1.5).set(
                ylabel="Optimality Ratio" if (fun_ix == 0) else None
            )

    fig.supxlabel(k_label)

    # All subplots' legends are the same, so turn off all but the last's
    # handles, labels = axs[0, -1].get_legend_handles_labels()
    # legend_proportion = (0.25 if plot_optimality_ratios else 0.30) * (1.4 if plot_our_bound else 1)
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0, 1 - legend_proportion, 1, legend_proportion), ncols=1)
    # for ax in axs[0, :]:
    #     ax.legend([], [], frameon=False)

    for ax in axs[0, 1:]:
        ax.legend([], [], frameon=False)
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(reversed(h), reversed(l), title='')

    fig.tight_layout()
    # fig.subplots_adjust(top=1-legend_proportion)
    # fig.subplots_adjust(right=.75)

    return fig


def inverse_monomial(deg):
    def f(x): return x**(-deg)
    f.degree = (0, deg)
    return f


if __name__ == "__main__":
    flamp.set_dps(300)  # compute with this many decimal digits precision
    print(f"Using {flamp.get_dps()} digits of precision")

    # sns.set(font_scale=2)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })
    # n_colors = 4
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,n_colors)))

    default_palette = np.array(sns.color_palette("rocket", 4))[[0, 2, 1, 3]]
    sns.set_palette(default_palette)
    # sns.set_palette(sns.color_palette("rocket"))

    dim = 100
    kappa = flamp.gmpy2.mpfr(100.)
    lambda_min = flamp.gmpy2.mpfr(1.)
    # a_diag_unif = mf.flipped_model_spectrum(dim, kappa, 1, lambda_min)
    a_diag_unif = flamp.linspace(lambda_min, kappa*lambda_min, dim)
    a_diag_geom = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
    a_diag_two_cluster = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=10, lambda_1=lambda_min)
    b = flamp.to_mp(mf.geometric_spectrum(dim, 1e4, 1000))

    # ks = list(range(1, dim//10)) + list(range(dim//10, dim-5, 5)) + list(range(dim-5, dim+1))
    ks = list(range(1, 61))
    # ks = list(range(1, 51, 10))
    # print("bAAAD FIX Me")

    zolotarev_13 = tylers_sqrt(13, float(a_diag_geom.min()), float(a_diag_geom.max()))
    print("general performance")
    sns.set_palette(np.array(sns.color_palette("rocket", 5))[[0, 3, 2, 1]])
    # Convergence of Lanczos on general functions
    general_performanace_funs = {
        r"$\mathbf A^{-2}\mathbf b$": inverse_monomial(2),
        r"$\exp(\mathbf A)b$": flamp.exp,
        r"$\log(\mathbf A) \mathbf b$": flamp.log
    }
    general_performanace_fig = convergence_superplot([a_diag_unif, a_diag_geom, a_diag_two_cluster], b, ks, general_performanace_funs, plot_our_bound=False, plot_spectrum_optimal=True, relative_error=True)
    general_performanace_fig.savefig('output/paper_plots/general_performance.svg')

    print("our bound")
    sns.set_palette(np.array(sns.color_palette("rocket", 5))[[0, 3, 1, 4]])
    # log_approx = baryrat.aaa(np.linspace(1, float(kappa), num=10_000), np.log, mmax=13, tol=1e-15)
    log_approx, _ = baryrat.brasil(flamp.log, (flamp.gmpy2.mpfr(1), flamp.gmpy2.mpfr(kappa)), 10, info=True)
    log_approx.degree = log_approx.degree()
    # Our bound vs convergence curve for rational functions
    our_bound_funs = {
        r"$\mathbf A^{-2}\mathbf b$": inverse_monomial(2),
        r"$r(\mathbf A)\mathbf b \approx \exp(\mathbf A) \mathbf b$ (deg=5)": exp_pade0_55,
        r"$r(\mathbf A)\mathbf b \approx \log(\mathbf A)\mathbf b$ (deg=10)": log_approx,
    }
    our_bound_fig = convergence_superplot([a_diag_unif, a_diag_geom, a_diag_two_cluster], b, ks, our_bound_funs, plot_our_bound=True, relative_error=True, plot_optimality_ratios=False)
    our_bound_fig.savefig('output/paper_plots/our_bound.svg')

    print("sqrt vs rat")
    sns.set_palette(sns.color_palette("rocket", 5))
    # Triangle inequality
    a_diag_10_outliers = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=10, lambda_1=lambda_min)
    fig_triangle, ax_triangle = plt.subplots(figsize=(5.4, 3.75))
    results = fun_vs_rationals(flamp.sqrt, a_diag_10_outliers, flamp.ones(dim), ks, [20, 15, 10, 5],
                               approximator=lambda degree: tylers_sqrt(degree, float(a_diag_geom.min()), float(a_diag_geom.max())),
                               fname="Square root",
                               relative_error=True)
    # results.to_csv("output/paper_data/triangle_inequality.csv")
    plot_convergence_curves(results[results.index <= 30], "Relative Error", r"Number of iterations ($k$)", ax=ax_triangle)
    ax_triangle.legend(title='')
    fig_triangle.tight_layout()
    fig_triangle.savefig('output/paper_plots/sqrt_vs_rat.svg')

    sns.set_palette(default_palette)
    # Indefinite spectrum
    def one_over_1plus_x_squared(x):
        return 1/(5 + x**2)
    one_over_1plus_x_squared.degree = (0, 2)

    def one_over_1minus_x_squared(x):
        return 1/(5 - x**2)
    one_over_1minus_x_squared.degree = (0, 2)

    print("indefinite")
    indefinite_fig = convergence_superplot(
        [np.hstack([-a_diag_geom, a_diag_geom])] * 3,
        flamp.ones(2 * dim), ks,
        {r"$\mathrm{sign}(\mathbf A)\mathbf b$": np.sign, r"$(5 - \mathbf A^2)^{-1} \mathbf b$": one_over_1minus_x_squared, r"$(5 + \mathbf A^2)^{-1} \mathbf b$": one_over_1plus_x_squared},
        plot_our_bound=False, relative_error=True, differentiate_sizes=False, plot_unif=False)
    indefinite_fig.savefig('output/paper_plots/indefinite.svg')

    sns.set_palette(sns.color_palette("rocket", 5))
    # optimal lower bound: C grows as sqrt(kappa * q)
    df_lower = pd.read_csv("output/optim_grid_search/2023-03-02_08:04.tsv", sep='\t')
    df_lower["log_kappa"] = np.log10(df_lower["kappa"])

    print("opt lower bound")
    fig_lower_bound, axs_lower_bound = plt.subplots(1, 2, figsize=(8, 4))
    p1 = sns.scatterplot(x='q', y='ratio_max', hue='log_kappa', data=df_lower, legend=False, palette=sns.color_palette("rocket", 5), ax=axs_lower_bound[0], s=60)
    p2 = sns.lineplot(x=df_lower.q.unique(), y=np.sqrt(df_lower.q.unique() * df_lower.kappa.max()), ax=axs_lower_bound[0], lw=1.5, ls=':')
    axs_lower_bound[0].set(xscale='log', yscale='log', xlabel=r"$q$", ylabel=r'Max Optimality Ratio ($C$)')
    axs_lower_bound[0].set_xscale('log', base=2)
    # axs_lower_bound[0].legend([r"Observed $C$", r"$\sqrt{q}$"])

    p1 = sns.scatterplot(x='kappa', y='ratio_max', hue='log_kappa', data=df_lower, legend=False, palette=sns.color_palette("rocket", 5), ax=axs_lower_bound[1], s=60)
    p2 = sns.lineplot(x=df_lower.kappa.unique(), y=np.sqrt(df_lower.kappa.unique() * df_lower.q.max()), ax=axs_lower_bound[1], lw=1.5, ls=':')
    axs_lower_bound[1].set(xscale='log', yscale='log', xlabel=r'$\kappa$', ylabel='')
    # axs_lower_bound[1].legend([r"Observed $C$", r"$\sqrt{\kappa}$"])
    fig_lower_bound.tight_layout()
    fig_lower_bound.savefig("output/paper_plots/opt_lower_bound.svg")

    # lower bound for lanczos-OR: C grows as sqrt(kappa ^ q)
    df_or = pd.read_csv("output/optim_grid_search/2023-03-03_03:08.tsv", sep='\t')
    df_or["log_kappa"] = np.log10(df_or["kappa"])

    print("lanczos or lower")
    fig_lower_bound, axs_lower_bound = plt.subplots(1, 2, figsize=(8, 4))
    p1 = sns.scatterplot(x='q', y='ratio_max', hue='log_kappa', data=df_or, legend=False, palette=sns.color_palette("rocket", 5), ax=axs_lower_bound[0], s=60)
    p2 = sns.lineplot(x=np.geomspace(df_or.q.min(), df_or.q.max()), y=(df_or.kappa.max() ** (np.geomspace(df_or.q.min(), df_or.q.max()) / 2)), ax=axs_lower_bound[0], lw=1.5, ls=':')
    axs_lower_bound[0].set(xscale='log', yscale='log', xlabel=r"$q$", ylabel=r'Max Optimality ratio ($C$)')
    axs_lower_bound[0].set_xscale('log', base=2)
    # axs_lower_bound[0].legend([r"Observed $C$", r"$\sqrt{q}$"])

    p1 = sns.scatterplot(x='kappa', y='ratio_max', hue='log_kappa', data=df_or, legend=False, palette=sns.color_palette("rocket", 5), ax=axs_lower_bound[1], s=60)
    p2 = sns.lineplot(x=df_or.kappa.unique(), y=(df_or.kappa.unique() ** (df_or.q.max() / 2)), ax=axs_lower_bound[1], lw=1.5, ls=':')
    axs_lower_bound[1].set(xscale='log', yscale='log', xlabel=r'$\kappa$', ylabel='')
    # axs_lower_bound[1].legend([r"Observed $C$", r"$\sqrt{\kappa}$"])
    fig_lower_bound.tight_layout()
    fig_lower_bound.savefig("output/paper_plots/lanczos_or_lower.svg")

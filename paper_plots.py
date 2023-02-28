import flamp
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

import baryrat

import matrix_functions as mf
from fa_performance import fa_performance, fun_vs_rationals
from zolotarev import tylers_sqrt, exp_pade0_55


def plot_convergence_curves(results, error_label, k_label, title=None, ax=None):
    str_error_label = "error" if error_label is None else error_label
    str_k_label = "k" if k_label is None else k_label

    results_long = results.reset_index(names=str_k_label).melt(
        id_vars=[str_k_label], value_name=str_error_label, var_name="Approximant")

    sns.lineplot(
        x=str_k_label,
        y=str_error_label,
        hue="Approximant",
        style="Approximant",
        # dashes=[],
        # marker="2",
        data=results_long,
        ax=ax,
    ).set(
        title=title,
        xlabel=k_label,
        ylabel=error_label,
        yscale='log'
    )


def convergence_superplot(a_diag, b, ks, functions_dict, relative_error, plot_our_bound=True, plot_optimality_ratios=True):
    fig_height = 4.8  # default. shouldn't matter when using svg
    row_heights = [1] + ([0.5] if plot_optimality_ratios else [])
    fig, axs = plt.subplots(
        len(row_heights), len(functions_dict),
        sharex=True,
        height_ratios=row_heights,
        figsize=(fig_height * len(functions_dict) * 1.3 + 20, sum(row_heights) * fig_height * 1.3),
        squeeze=False
    )

    if relative_error:
        error_label = "Relative Error"
    else:
        error_label = "Error"
    k_label = "Number of matrix-vector products ($k$)"

    for fun_ix, (fun_label, fun) in enumerate(tqdm(functions_dict.items())):
        results = fa_performance(fun, a_diag, b, ks,
                                 relative_error=relative_error,
                                 our_bound=plot_our_bound)

        plot_convergence_curves(
            results,
            error_label=error_label if (fun_ix == 0) else None,
            k_label=None,
            title=fun_label,
            ax=axs[0, fun_ix]
        )

        if plot_optimality_ratios:
            krylov_optimal_label = r"$||\mathrm{opt}_k(I) - f(A)b||_2$"
            lanczos_label = r"$||\mathrm{lan}_k - f(A)b||_2$"
            optimality_ratios = results[lanczos_label] / results[krylov_optimal_label]
            sns.lineplot(data=optimality_ratios, ax=axs[1, fun_ix]).set(
                ylabel="Optimality Ratio" if (fun_ix == 0) else None
            )

    fig.suptitle("Approximation of $f(A)b$")  # should we say spectrum is this, $b = that$
    fig.supxlabel(k_label)

    # All subplots' legends are the same, so turn off all but the last's
    for ax in axs[0, :-1]:
        ax.legend([], [], frameon=False)
    axs[0, -1].legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)

    return fig


def plot_fun_vs_rationals(results, uniform_errors, error_label, ax=None):
    str_k_label = "k"
    str_error_label = "error" if error_label is None else error_label

    results_long = results.reset_index(names=str_k_label).melt(
        id_vars=[str_k_label], value_name=str_error_label, var_name="Approximant")

    sns.lineplot(
        x=str_k_label,
        y=str_error_label,
        hue="Approximant",
        style="Approximant",
        data=results_long,
        ax=ax,
    )
    # for k, uniform_error in uniform_errors.items():
    #     plt.
    print(uniform_errors.astype(float))
    ax.set(
        title="sooomeeething",
        xlabel=str_k_label,
        ylabel=error_label,
        yscale='log'
    )


def inverse_monomial(deg):
    def f(x): return x**(-deg)
    f.degree = (0, deg)
    return f


if __name__ == "__main__":
    flamp.set_dps(300)  # compute with this many decimal digits precision
    print(f"Using {flamp.get_dps()} digits of precision")

    sns.set(font_scale=1.5)

    dim = 100
    kappa = flamp.gmpy2.mpfr(100.)
    lambda_min = flamp.gmpy2.mpfr(1.)
    a_diag = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
    b = flamp.ones(dim)

    # ks = list(range(1, dim//10)) + list(range(dim//10, dim-5, 5)) + list(range(dim-5, dim+1))
    ks = list(range(1, 51))

    results, uniform_errors = fun_vs_rationals(flamp.sqrt, a_diag, b, ks, [2, 4, 8, 16, 24], relative_error=True)
    print(results.loc[50].astype(float))
    print((uniform_errors * flamp.sqrt(uniform_errors.index * kappa)).astype(float))
    fig1, ax1 = plt.subplots()
    plot_fun_vs_rationals(results[results.index <= 50], uniform_errors, error_label="Relative Error", ax=ax1)
    fig1.savefig('output/sqrt_vs_rat.svg')


    # zolotarev_13 = tylers_sqrt(13, float(a_diag.min()), float(a_diag.max()))
    # brasil_sqrt_13, brasil_sqrt_13_info = baryrat.brasil(flamp.sqrt, (a_diag.min(), a_diag.max()), 13, info=True)
    # brasil_sqrt_13.degree = brasil_sqrt_13.degree()
    # brasil_exp_13, brasil_exp_13_info = baryrat.brasil(flamp.exp, (a_diag.min(), a_diag.max()), 13, info=True)
    # brasil_exp_13.degree = brasil_exp_13.degree()

    # name2function = {
    #     r"$A^{-2}b$": inverse_monomial(2),
    #     r"$e^Ab$": flamp.exp,
    #     r"$\sqrt{A}b$": flamp.sqrt,
    #     f"$\\mathrm{{zolotarev}}_{{{zolotarev_13.degree[1]}}}(A)b$": zolotarev_13,
    #     "5,5-Pade of exp about 0": exp_pade0_55,
    #     "Rational approx to sqrt (deg=13)": brasil_sqrt_13,
    #     "Rational approx to exp (deg=13)": brasil_exp_13
    # }

    # general_performanace_funs = {
    #     r"$A^{-2}b$": inverse_monomial(2),
    #     r"$e^Ab$": flamp.exp,
    #     r"$\sqrt{A}b$": flamp.sqrt
    # }
    # general_performanace_fig = convergence_superplot(a_diag, b, ks, general_performanace_funs, plot_our_bound=False, relative_error=True)
    # general_performanace_fig.savefig('output/general_performance.svg')

    # our_bound_funs = {
    #     r"$A^{-2}b$": inverse_monomial(2),
    #     "5,5-Pade of exp about 0": exp_pade0_55,
    #     r"$\\mathrm{{zolotarev}}_{{13}}(A)b$": tylers_sqrt(13, float(a_diag.min()), float(a_diag.max())),
    # }
    # our_bound_fig = convergence_superplot(a_diag, b, ks, our_bound_funs, plot_our_bound=True, relative_error=True, plot_optimality_ratios=False)
    # our_bound_fig.savefig('output/our_bound.svg')



    # # fig2_funs = {name: name2function[name] for name in [r"$A^{-2}b$",  f"$\\mathrm{{zolotarev}}_{{{zolotarev_degree}}}(A)b$"]}  # r"$e^Ab$",
    # fig2_funs = fig1_funs
    # fig2 = convergence_superplot(a_diag, b, ks, fig2_funs, relative_error=True, plot_optimality_ratios=False)
    # fig2.savefig('output/our_bound.svg')

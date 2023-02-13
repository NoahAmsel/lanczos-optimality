import flamp
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

import matrix_functions as mf
from fa_performance import fa_performance
from zolotarev import tylers_sqrt


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
        data=results_long,
        ax=ax,
    ).set(
        title=title,
        xlabel=k_label,
        ylabel=error_label,
        yscale='log'
    )


def curves(a_diag, b, ks, functions_dict, relative_error, plot_our_bound=True, plot_optimality_ratios=True):
    fig_height = 4.8  # default. shouldn't matter when using svg
    row_heights = [1, 0.3] if plot_optimality_ratios else [1]
    fig, axs = plt.subplots(
        len(row_heights), len(functions_dict),
        sharex=True,
        height_ratios=row_heights,
        figsize=(fig_height * (len(functions_dict)), sum(row_heights) * fig_height),
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

    return fig


def inverse_monomial(deg):
    def f(x): return x**(-deg)
    f.degree = (0, deg)
    return f


if __name__ == "__main__":
    flamp.set_dps(350)  # compute with this many decimal digits precision
    print(f"Using {flamp.get_dps()} digits of precision")

    dim = 100
    kappa = flamp.gmpy2.mpfr(100.)
    lambda_min = flamp.gmpy2.mpfr(1.)
    a_diag = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
    b = flamp.ones(dim)

    # ks = list(range(1, dim//10)) + list(range(dim//10, dim-5, 5)) + list(range(dim-5, dim+1))
    ks = list(range(1, dim//10)) + list(range(dim//10, dim, 5))

    zolotarev_degree = 13
    name2function = {
        r"$A^{-2}b$": inverse_monomial(2),
        r"$e^Ab$": flamp.exp,
        r"$\sqrt{A}b$": flamp.sqrt,
        f"$\\mathrm{{zolotarev}}_{{{zolotarev_degree}}}(A)b$": tylers_sqrt(
            zolotarev_degree, float(a_diag.min()), float(a_diag.max()))
    }

    fig1_funs = {name: name2function[name] for name in [r"$A^{-2}b$", r"$e^Ab$", r"$\sqrt{A}b$"]}
    fig1 = curves(a_diag, b, ks, fig1_funs, plot_our_bound=False, relative_error=True)
    fig1.savefig('output/lanczos_performance.svg')

    fig2_funs = {name: name2function[name] for name in [
        r"$A^{-2}b$",  f"$\\mathrm{{zolotarev}}_{{{zolotarev_degree}}}(A)b$"]}  # r"$e^Ab$",
    fig2 = curves(a_diag, b, ks, fig2_funs, relative_error=True, plot_optimality_ratios=False)
    fig2.savefig('output/our_bound.svg')

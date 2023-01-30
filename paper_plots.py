import flamp
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

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


def curves(a_diag, b, ks, functions_dict, relative_error, denom_degrees_dict=None, plot_optimality_ratios=True):
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

    for fun_ix, (fun_label, fun) in enumerate(functions_dict.items()):
        print(f"Plotting {fun_label}...")
        if (denom_degrees_dict is not None) and (denom_degrees_dict[fun_label] > 0):
            denom_degree = denom_degrees_dict[fun_label]
        else:
            denom_degree = None
        results = fa_performance(fun, a_diag, b, ks, denom_degree=denom_degree, relative_error=relative_error)

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

    functions_table = pd.DataFrame(
        index=[
            r"$A^{-2}b$",
            r"$e^Ab$",
            r"$\sqrt{A}b$",
            f"$\\mathrm{{zolotarev}}_{{{zolotarev_degree}}}(A)b$"
        ],
        data={
            "function": [
                lambda x: x**(-2),
                flamp.exp,
                flamp.sqrt,
                tylers_sqrt(zolotarev_degree, float(a_diag.min()), float(a_diag.max()))
            ],
            "denom_degree": [2, -1, -1, zolotarev_degree]
        }
    )

    fig1_curves = functions_table.loc[[r"$A^{-2}b$", r"$e^Ab$", r"$\sqrt{A}b$"]]
    fig1 = curves(a_diag, b, ks, fig1_curves["function"], relative_error=True)
    fig1.savefig('output/lanczos_performance.svg')

    fig2_curves = functions_table.loc[[r"$A^{-2}b$", r"$e^Ab$", f"$\\mathrm{{zolotarev}}_{{{zolotarev_degree}}}(A)b$"]]
    fig1 = curves(
        a_diag, b, ks, fig2_curves["function"],
        relative_error=True, denom_degrees_dict=fig2_curves["denom_degree"],
        plot_optimality_ratios=False)
    fig1.savefig('output/our_bound.svg')

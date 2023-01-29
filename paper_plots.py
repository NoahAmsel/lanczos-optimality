import flamp
import seaborn as sns
from matplotlib import pyplot as plt

import matrix_functions as mf
from fa_performance import fa_performance

flamp.set_dps(50)  # compute with this many decimal digits precision

dim = 100
kappa = flamp.gmpy2.mpfr(1_000.)
lambda_min = flamp.gmpy2.mpfr(1.)
a_diag = mf.geometric_spectrum(dim, kappa, rho=1e-5, lambda_1=lambda_min)
b = flamp.ones(dim)

# ks = list(range(1, dim//10)) + list(range(dim//10, dim-5, 5)) + list(range(dim-5, dim+1))
ks = list(range(1, dim//10)) + list(range(dim//10, dim, 5))

functions_dict = {
    "$A^{-2}b$": lambda x: x**(-2),
    "$e^Ab$": flamp.exp,
    "$\sqrt{A}b$": flamp.sqrt,
}

fig_height = 4.8  # default. shouldn't matter when using svg
fig, axes = plt.subplots(
    1, len(functions_dict),
    figsize=(fig_height * (len(functions_dict)), fig_height)
)

relative_error = True
if relative_error:
    error_label = "Relative Error"
else:
    error_label = "Error"
k_label = "Number of matrix-vector products ($k$)"

for (fun_label, fun), ax in zip(functions_dict.items(), axes):
    print(f"Plotting {fun_label}...")
    results = fa_performance(fun, a_diag, b, ks)
    krylov_label = "||\mathrm{opt}_k(I) - f(A)b||_2"

    # notice that it's relative to the *Euclidean* norm of the ground truth
    if relative_error:
        results /= mf.norm(mf.diagonal_fa(fun, a_diag, b))

    # TODO: debug this
    results = results.astype(float)

    results_long = results.reset_index(names=k_label).melt(
        id_vars=[k_label], value_name=error_label, var_name="Approximant")

    sns.lineplot(
        x=k_label,
        y=error_label,
        hue="Approximant",
        style="Approximant",
        data=results_long,
        ax=ax,
    ).set(
        title=fun_label,
        xlabel=None,
        ylabel=None,
        yscale='log'
    )

fig.suptitle("Approximation of $f(A)b$, $b = 1$")
fig.supxlabel(k_label)
fig.supylabel(error_label)

# All subplots' legends are the same, so turn off all but first's
for ax in axes[1:]:
    ax.legend([], [], frameon=False)

plt.savefig('output/lanczos_performance.svg')
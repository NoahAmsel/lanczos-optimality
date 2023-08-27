import flamp
import numpy as np
import seaborn as sns


class InversePolynomial:
    def __init__(self, polynomial): self.polynomial = polynomial
    def __call__(self, x): return 1/self.polynomial(x)
    def poles(self): return self.polynomial.roots()
    def degree_denom(self): return self.polynomial.degree()


class InverseMonomial(InversePolynomial):
    def __init__(self, deg): super().__init__(np.polynomial.Polynomial.fromroots([0] * deg))


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
    # Lines should be given a z-ordering with the first one at the bottom. so reverse the order of the columns
    error_df_long = error_df.iloc[:, ::-1].reset_index(names=k_label).melt(
        id_vars=[k_label], value_name=error_label, var_name="Line"
    )

    if "title" in kwargs:
        title = kwargs.pop("title")
    else:
        title = None

    ax = sns.lineplot(
        x=k_label,
        y=error_label,
        data=error_df_long,
        hue="Line",
        style="Line",
        size="Line" if ("sizes" in kwargs) else None,
        **kwargs
    )
    ax.set(
        title=title,
        xlabel=k_label,
        ylabel=error_label,
        yscale='log'
    )
    ax.legend(title='')
    handles, labels = ax.get_legend_handles_labels()
    # Undo the reversal above; lines should appear in the legend in the order given by user.
    ax.legend(reversed(handles), reversed(labels), title='')
    return ax

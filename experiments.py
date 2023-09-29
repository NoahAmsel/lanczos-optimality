import flamp
import numpy as np
from scipy.optimize import minimize_scalar
import seaborn as sns

import matrix_functions as mf


class InversePolynomial:
    def __init__(self, polynomial): self.polynomial = polynomial
    def __call__(self, x): return 1/self.polynomial(x)
    def poles(self): return self.polynomial.roots()
    def degree_denom(self): return self.polynomial.degree()


class InverseMonomial(InversePolynomial):
    def __init__(self, deg): super().__init__(np.polynomial.Polynomial.fromroots([0] * deg))


class ExpRationalApprox:
    def __init__(self, a, b, t, degree, num_grid=None):
        self.degree = degree
        if num_grid is None:
            num_grid = degree * 100
        x_grid = flamp.linspace(a, b, num_grid)
        transformed_grid = 1/(1 + x_grid/self.degree)
        self.interval = (1/(1 + b/self.degree), 1/(1 + a/self.degree))
        V = mf.cheb_vandermonde(transformed_grid, self.degree, interval=self.interval)
        f_grid = flamp.exp(t*x_grid)
        self.coeff = flamp.qr_solve(V, f_grid)

    def __call__(self, x):
        transformed_x = 1/(1 + x/self.degree)
        VV = mf.cheb_vandermonde(transformed_x, self.degree, interval=self.interval)
        return VV @ self.coeff

    def degree_denom(self): return self.degree
    def poles(self): return flamp.to_mp([-self.degree])


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


def worst_b0(f, spectrum, ks, bounds, norm_matrix_sqrt=None, xatol=1e-10):
    b = flamp.ones(len(spectrum))

    def objective(b0):
        b[0] = flamp.gmpy2.mpfr(b0)
        problem = mf.DiagonalFAProblem(f, spectrum, b)
        lan = np.array([problem.lanczos_error(k, norm_matrix_sqrt=norm_matrix_sqrt) for k in ks])
        opt = np.array([problem.instance_optimal_error(k, norm_matrix_sqrt=norm_matrix_sqrt) for k in ks])
        # negative because we use a minimize function to solve a maximization problem
        return -float((lan / opt).max())

    res = minimize_scalar(objective, bounds=bounds, options=dict(xatol=xatol))
    return res.x, -res.fun


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

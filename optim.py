from datetime import datetime

import flamp
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from tqdm.contrib.itertools import product

import seaborn as sns
import matplotlib.pyplot as plt

import matrix_functions as mf
from fa_performance import fa_performance

flamp.set_dps(300)  # compute with this many decimal digits precision


def inverse_monomial(deg):
    def f(x): return x**(-deg)
    f.degree = (0, deg)
    return f




# Fixed
dim = 100
lambda_min = flamp.gmpy2.mpfr(1.)
ks = list(range(1, 20))  # size of Krylov subspace

use_Anorm = False
if use_Anorm:
    krylov_optimal_label = r"$||\mathrm{opt}_k(A) - f(A)b||_A$"
    lanczos_label = r"$||\mathrm{lan}_k - f(A)b||_A$"
else:
    krylov_optimal_label = r"$||\mathrm{opt}_k(I) - f(A)b||_2$"
    lanczos_label = r"$||\mathrm{lan}_k - f(A)b||_2$"


def optimal_b0(f, a_diag, ks, bounds, xatol=1e-10):
    def objective(b0):
        b = flamp.ones(dim)
        b[0] = flamp.gmpy2.mpfr(b0)
        results = fa_performance(f, a_diag, b, ks, relative_error=True,
                                 uniform_bound_interpolation=False,
                                 lanczos=(not use_Anorm),
                                 lanczos_Anorm=use_Anorm,
                                 krylov_optimal=(not use_Anorm),
                                 krylov_optimal_Anorm=use_Anorm,
                                 our_bound=False)
        optimality_ratios = (results[lanczos_label] / results[krylov_optimal_label]).astype(float)
        # IMPORTANT: we want to minimize
        return -optimality_ratios.max()

    res = minimize_scalar(objective, bounds=bounds, options=dict(xatol=xatol))
    return res.x, -res.fun


# high_cluster_width = np.geomspace(0.5e-5, 0.5e0, num=20)[4]
# results = []
# for kappa, q in tqdm(product([10**2, 10**3, 10**4, 10**5, 10**6], range(2, 40, 2))):
#     kappa = flamp.gmpy2.mpfr(kappa)

#     a_diag = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=1,
#                                      low_cluster_width=0.005,
#                                      high_cluster_width=high_cluster_width)

#     f = inverse_monomial(q)

#     opt_b0, opt_ratio = optimal_b0(f, a_diag, ks, (1e-8, 1))

#     result = dict(
#         kappa=kappa,
#         q=q,
#         width=high_cluster_width,
#         dimension=dim,
#         b0=opt_b0,
#     )
#     if use_Anorm:
#         result['A_norm_ratio_max'] = opt_ratio
#     else:
#         result['ratio_max'] = opt_ratio
#     results.append(result)

# df = pd.DataFrame(results)
# df.to_csv(f"output/optim_grid_search/{datetime.today():%Y-%m-%d_%H:%M}.tsv", sep='\t', index=False)

df = pd.read_csv("output/optim_grid_search/2023-02-26_15:55.tsv", sep='\t')
sns.heatmap(df.pivot(index="kappa", columns="q", values="ratio_max"), annot=True)

# Plot how C growss as sqrt(kappa * q)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
p1 = sns.scatterplot(x='q', y='ratio_max', data=df, ax=axs[0])
p2 = sns.lineplot(x=df.q.unique(), y=np.sqrt(df.q.unique() * df.kappa.max()), ax=axs[0])
axs[0].set(xscale='log', yscale='log', ylabel=r'Optimality ratio ($C$)')
axs[0].legend([r"Observed $C$", r"$\sqrt{q}$"], loc='lower right')

p1 = sns.scatterplot(x='kappa', y='ratio_max', data=df, ax=axs[1])
p2 = sns.lineplot(x=df.kappa.unique(), y=np.sqrt(df.kappa.unique() * df.q.max()), ax=axs[1])
axs[1].set(xscale='log', yscale='log', xlabel=r'$\kappa$', ylabel='')
axs[1].legend([r"Observed $C$", r"$\sqrt{\kappa}$"], loc='lower right')
fig.suptitle(r"For $f(x) = x^q$, the optimality ratio $C = \Omega(\sqrt{q\kappa})$")
fig.savefig("output/opt_lower_bound.svg")

###############
# Investigating periodicity
###############
# kappa = flamp.gmpy2.mpfr(100)
# f = inverse_monomial(2)
# width = np.geomspace(0.5e-5, 0.5e0, num=40)[20]
# a_diag = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=1, low_cluster_width=0.005, high_cluster_width=width)
# approx_peaks = np.geomspace(1e-8, 1e3, num=60)[[1, 19, 37, 54]]

# fig, axs = plt.subplots(len(approx_peaks), 1)

# for ix, approx_peak in enumerate(approx_peaks):
#     opt_b0, _ = optimal_b0(f, a_diag, ks, (approx_peak / 2, approx_peak * 2))
#     b = flamp.ones(dim)
#     b[0] = flamp.gmpy2.mpfr(opt_b0)
#     results = fa_performance(f, a_diag, b, ks, relative_error=True,
#                              uniform_bound_interpolation=False,
#                              lanczos=(not use_Anorm),
#                              lanczos_Anorm=use_Anorm,
#                              krylov_optimal=(not use_Anorm),
#                              krylov_optimal_Anorm=use_Anorm,
#                              our_bound=False)
#     optimality_ratios = results[lanczos_label] / results[krylov_optimal_label]
#     sns.lineplot(data=optimality_ratios, ax=axs[ix]).set(
#         xlabel="Number of matrix-vector products ($k$)",
#         ylabel="Optimality Ratio",
#         title=str(approx_peak)
#     )

# fig.savefig("output/approx_peak.svg")

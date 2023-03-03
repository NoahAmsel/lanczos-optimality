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
from fa_performance import fa_performance, lanczos_error_curve

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
use_alt_norm = False
if use_Anorm:
    krylov_optimal_label = r"$||\mathrm{opt}_k(A) - f(A)b||_A$"
    lanczos_label = r"$||\mathrm{lan}_k - f(A)b||_A$"
else:
    krylov_optimal_label = r"$||\mathrm{opt}_k(I) - f(A)b||_2$"
    lanczos_label = r"$||\mathrm{lan}_k - f(A)b||_2$"
if use_alt_norm:
    krylov_optimal_label = r"$||\mathrm{opt}_k(A) - f(A)b||_{A^q}$"
    lanczos_label = r"$||\mathrm{lan}_k - f(A)b||_{A^q}$"


def optimal_b0(f, a_diag, ks, bounds, alt_norm=None, xatol=1e-10):
    def objective(b0):
        b = flamp.ones(dim)
        b[0] = flamp.gmpy2.mpfr(b0)
        results = fa_performance(f, a_diag, b, ks, relative_error=True,
                                 uniform_bound_interpolation=False,
                                 lanczos=((not use_Anorm) and (alt_norm is None)),
                                 lanczos_Anorm=(use_Anorm and (alt_norm is None)),
                                 lanczos_alt_norms=alt_norm,
                                 krylov_optimal=((not use_Anorm) and (alt_norm is None)),
                                 krylov_optimal_Anorm=(use_Anorm and (alt_norm is None)),
                                 krylov_optimal_alt_norms=alt_norm,
                                 our_bound=False)
        optimality_ratios = (results[lanczos_label] / results[krylov_optimal_label]).astype(float)
        # IMPORTANT: we want to minimize
        return -optimality_ratios.max()

    res = minimize_scalar(objective, bounds=bounds, options=dict(xatol=xatol))
    return res.x, -res.fun

high_cluster_width = np.geomspace(0.5e-5, 0.5e0, num=20)[4]
results = []
for kappa, q in tqdm(product([10**2, 10**3, 10**4, 10**5, 10**6], [2, 4, 8, 16, 32, 64])):
    kappa = flamp.gmpy2.mpfr(kappa)

    a_diag = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=1,
                                     low_cluster_width=0.005,
                                     high_cluster_width=high_cluster_width)

    f = inverse_monomial(q)

    alt_norm = {"{A^q}": mf.DiagonalMatrix(flamp.sqrt(a_diag ** q))}
    opt_b0, opt_ratio = optimal_b0(f, a_diag, ks, (1e-8, 1), alt_norm=(alt_norm if use_alt_norm else None))

    result = dict(
        kappa=kappa,
        q=q,
        width=high_cluster_width,
        dimension=dim,
        b0=opt_b0,
    )
    if use_Anorm:
        result['A_norm_ratio_max'] = opt_ratio
    else:
        result['ratio_max'] = opt_ratio
    results.append(result)

df = pd.DataFrame(results)
df.to_csv(f"output/optim_grid_search/{datetime.today():%Y-%m-%d_%H:%M}.tsv", sep='\t', index=False)

# sns.heatmap(df.pivot(index="kappa", columns="q", values="ratio_max"), annot=True)


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

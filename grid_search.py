from datetime import datetime

import flamp
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.itertools import product

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

# kappas = [flamp.gmpy2.mpfr(10**2), flamp.gmpy2.mpfr(10**4), flamp.gmpy2.mpfr(10**6)]
# widths = np.geomspace(0.5e-5, 0.5e-3, num=10)
# b0s = np.geomspace(1e-8, 1e3, num=40)
# qs = [2, 3, 4, 6, 8]
kappas = [flamp.gmpy2.mpfr(10**2)]
widths = [np.geomspace(0.5e-5, 0.5e0, num=40)[20]]
b0s = np.geomspace(1e-8, 1e3, num=60)
qs = [2]
data = []
# for high_cluster_width, b0, q in [(0.5, 1, 2), (0.005, 0.001, 6)]:
for kappa, high_cluster_width, b0, q in tqdm(product(kappas, widths, b0s, qs)):
    a_diag = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=1,
                                     low_cluster_width=0.005,
                                     high_cluster_width=high_cluster_width)

    b = flamp.ones(dim)
    b[0] = flamp.gmpy2.mpfr(b0)

    f = inverse_monomial(q)

    results = fa_performance(f, a_diag, b, ks, relative_error=True,
                             uniform_bound_interpolation=False,
                             lanczos=(not use_Anorm),
                             lanczos_Anorm=use_Anorm,
                             krylov_optimal=(not use_Anorm),
                             krylov_optimal_Anorm=use_Anorm,
                             our_bound=False)
    optimality_ratios = (results[lanczos_label] / results[krylov_optimal_label]).astype(float)
    data.append((
        kappa,
        high_cluster_width,
        b0,
        q,
        optimality_ratios.max(),
        optimality_ratios.index[optimality_ratios.argmax()],
        optimality_ratios[optimality_ratios.index.max()]))

labels = ["ratio_max", "ratio_argmax", "ratio_last"]
if use_Anorm:
    labels = [f"A_norm_{label}" for label in labels]
results = pd.DataFrame(data, columns=["kappa", "width", "b0", "q"] + labels)
results["kappa"] = float(kappa)
results["dimension"] = dim
results.to_csv(f"output/grid_search/{datetime.today():%Y-%m-%d_%H:%M}.tsv", sep='\t', index=False)

# Findings so far
# max ratio overall is about 173, but a lot of examples come close to that
# max ratio is definitely an increasing function of q
# it's lower (~100) for big values of b0, but pretty constant throughout most of the range
# it's a little lower for big widths, but not that much
# with kappa = 10^2, the highest ratio was 7.81. with kappa = 10^4 it was 78.7
# these are mighty close to 10 and 10^2, ie sqrt(kappa)

# setting width = np.geomspace(0.5e-5, 0.5e0, num=40)[20], kappa = 100, q = 2,
# the peaks are around np.geomspace(1e-8, 1e3, num=60)[[ 1, 19, 37, 54]]
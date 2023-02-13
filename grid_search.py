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
kappa = flamp.gmpy2.mpfr(10_000.)
lambda_min = flamp.gmpy2.mpfr(1.)
ks = list(range(1, 20))  # size of Krylov subspace

krylov_optimal_label = r"$||\mathrm{opt}_k(I) - f(A)b||_2$"
lanczos_label = r"$||\mathrm{lan}_k - f(A)b||_2$"

# widths = np.geomspace(0.5e-5, 0.5e0, num=20)
# b0s = np.geomspace(1e-8, 1e3, num=40)
# qs = [2, 3, 4, 6, 8]
widths = np.geomspace(0.5e-5, 0.5e0, num=40)
b0s = np.geomspace(1e-8, 1e3, num=60)
qs = [2]
data = []
# for high_cluster_width, b0, q in [(0.5, 1, 2), (0.005, 0.001, 6)]:
for high_cluster_width, b0, q in tqdm(product(widths, b0s, qs)):
    a_diag = mf.two_cluster_spectrum(dim, kappa, low_cluster_size=1,
                                     low_cluster_width=0.005,
                                     high_cluster_width=high_cluster_width)

    b = flamp.ones(dim)
    b[0] = flamp.gmpy2.mpfr(b0)

    f = inverse_monomial(q)

    results = fa_performance(f, a_diag, b, ks, relative_error=True, uniform_bound_interpolation=False, our_bound=False)
    optimality_ratios = (results[lanczos_label] / results[krylov_optimal_label]).astype(float)
    data.append((
        high_cluster_width,
        b0,
        q,
        optimality_ratios.max(),
        optimality_ratios.index[optimality_ratios.argmax()],
        optimality_ratios[optimality_ratios.index.max()]))

results = pd.DataFrame(data, columns=["width", "b0", "q", "ratio_max", "ratio_argmax", "ratio_last"])
results["kappa"] = float(kappa)
results["dimension"] = dim
results.to_csv(f"output/grid_search/{datetime.today():%Y-%m-%d_%H:%M}.tsv", sep='\t', index=False)

# Findings so far
# max ratio overall is about 173, but a lot of examples come close to that
# max ratio is definitely an increasing function of q
# it's lower (~100) for big values of b0, but pretty constant throughout most of the range
# it's a little lower for big widths, but not that much

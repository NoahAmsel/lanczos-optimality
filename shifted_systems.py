import flamp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from fa_performance import fa_performance

flamp.set_dps(300)  # compute with this many decimal digits precision
print(f"Using {flamp.get_dps()} digits of precision")


def reciprocal(x): return x**(-1)


dim = 50
kappa = flamp.gmpy2.mpfr(2.)
lambda_min = flamp.gmpy2.mpfr(1.)
spectrum = flamp.linspace(lambda_min, kappa*lambda_min, dim)
b = flamp.ones(dim)
ks = list(range(1, 42, 5))

options = dict(relative_error=False, lanczos=True,
               uniform_bound_interpolation=False, krylov_optimal=False, our_bound=False)
sum_of_min = fa_performance(reciprocal, spectrum, b, ks, **options) + fa_performance(reciprocal, spectrum + 200., b, ks, **options)
min_of_sum = fa_performance(reciprocal, np.hstack((spectrum, spectrum + 200.)), np.hstack((b, b)), ks, **options)
df = pd.DataFrame({
    "sum of min": sum_of_min[sum_of_min.columns[0]],
    "min of sum": min_of_sum[min_of_sum.columns[0]],
})

plt.semilogy(df.index, df["sum of min"], df.index, df["min of sum"])
plt.legend(["sum of min", "min of sums"])

import numpy as np
import numpy.linalg as lin
from scipy.special import lambertw

from .utils import exp, linspace, log, norm


def generate_orthonormal(n, seed=None):
    M = np.random.default_rng(seed).normal(size=(n, n))
    Q, _ = lin.qr(M)
    return Q


def generate_symmetric(eigenvalues, seed=None):
    Q = generate_orthonormal(len(eigenvalues), seed=seed)
    return Q @ np.diag(eigenvalues) @ Q.T


def model_spectrum(n, kappa, rho, lambda_1=1.):
    assert 0 < rho <= 1
    lambda_n = kappa * lambda_1
    gap = lambda_n - lambda_1
    return lambda_1 + (linspace(0, n-1, num=n) / (n-1)) * gap * (rho ** linspace(n-1, 0, num=n))


def flipped_model_spectrum(n, kappa, rho, lambda_1=1.):
    assert 0 < rho
    lambda_n = kappa * lambda_1
    gap = lambda_n - lambda_1
    return lambda_1 + gap * np.real(lambertw(rho * linspace(0, n-1, num=n)) / lambertw(rho * (n-1)))


def geometric_spectrum(n, kappa, rho, lambda_1=1.):
    # when rho < 1, there are a few small elements and many large ones
    # when rho > 1, there are many small elements and a few large ones
    assert 0 < rho 
    # this ranges from 0 to 1
    steps = (1 - (rho ** linspace(0, 1, n))) / (1 - rho)
    #                 this ranges from 1 to kappa
    return lambda_1 * (1 + steps * (kappa - 1))


def two_cluster_spectrum(n, kappa, low_cluster_size=1, low_cluster_width=0.03, high_cluster_width=None, lambda_1=1.):
    if high_cluster_width is None:
        high_cluster_width = low_cluster_width

    lambda_n = kappa * lambda_1
    gap = lambda_n - lambda_1
    low_cluster = linspace(lambda_1, lambda_1 + gap * low_cluster_width, num=low_cluster_size)
    high_cluster = linspace(lambda_n - gap * high_cluster_width, lambda_n, num=(n - low_cluster_size))
    return np.hstack([low_cluster, high_cluster])


def start_vec(eigenvalues, ritz_values):
    n = len(eigenvalues)
    assert len(ritz_values) == n - 1

    # TODO: sort eigenvalues and ritz values before the loop, so that
    # you're adding up all the small things first.
    # Also combine the two loops for the same reason

    # in case both dtypes are integral, the 1.0 turns it into float
    twice_log_p = 1.0 * np.zeros(n, dtype=np.result_type(eigenvalues, ritz_values))  # should we match data type of lam?
    for shift in range(1, n):
        twice_log_p -= log(np.abs(eigenvalues - np.roll(eigenvalues, shift)))
    for j in range(n - 1):
        twice_log_p -= log(np.abs(eigenvalues - ritz_values[j]))
    twice_log_p -= twice_log_p.max()  # for numerical reasons
    p = exp(twice_log_p / 2)
    return p / norm(p)

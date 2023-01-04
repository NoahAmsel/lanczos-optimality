import numpy as np
import numpy.linalg as lin
from scipy.special import lambertw

from .utils import linspace


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


def two_cluster_spectrum(n, kappa, low_cluster_size=1, low_cluster_width=0.03, high_cluster_width=None, lambda_1=1.):
    if high_cluster_width is None:
        high_cluster_width = low_cluster_width

    lambda_n = kappa * lambda_1
    gap = lambda_n - lambda_1
    low_cluster = linspace(lambda_1, lambda_1 + gap * low_cluster_width, num=low_cluster_size)
    high_cluster = linspace(lambda_n - gap * high_cluster_width, lambda_n, num=(n - low_cluster_size))
    return np.hstack([low_cluster, high_cluster])

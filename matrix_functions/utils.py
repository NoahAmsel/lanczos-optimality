import numpy as np
import numpy.linalg as lin


def generate_orthonormal(n, seed=None):
    M = np.random.default_rng(seed).normal(size=(n, n))
    Q, _ = lin.qr(M)
    return Q


def generate_symmetric(eigenvalues, seed=None):
    Q = generate_orthonormal(len(eigenvalues), seed=seed)
    return Q @ np.diag(eigenvalues) @ Q.T


def generate_model_spectrum(n, kappa, rho, lambda_1=1., seed=None):
    assert 0 < rho < 1
    lambda_n = kappa * lambda_1
    gap = lambda_n - lambda_1
    return lambda_1 + (np.linspace(0, n-1, num=n) / (n-1)) * gap * (rho ** np.linspace(n-1, 0, num=n))


def generate_outlier_spectrum(n, kappa, lambda_1=1.):
    eigenvalues = np.full(n, lambda_1)
    eigenvalues[-1] = kappa * lambda_1
    return eigenvalues

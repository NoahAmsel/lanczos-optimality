from functools import cache

import flamp
import numpy as np
import scipy.special as sps

from .problem import DiagonalFAProblem
from .utils import norm


def inv_sqrt_weights_poles(q, a, b):
    # TODO: should we change uses of np to flamp?
    # TODO: when a,b are extraprecision, see below
    # https://mpmath.org/doc/current/functions/elliptic.html?highlight=elliptic#jacobi-elliptic-functions

    # In the sidford paper, the approximation is valid in the range [gamma, 1]
    # that corresponds to [min(abs(lam)), max(abs(lam))] / max(abs(lam))
    # since sidford's gamma =tyler's sqrt(k2)

    assert 0 < a < b
    a = a
    b = b

    k2 = a/b  # this is called gamma^2 in the Sidford paper

    # From Hale page 11: "For M/m > 10^6 however, this approach can be unstable and better numerical stability..."
    Kp = sps.ellipk(1-k2)  # this is called K prime in the Sidford paper

    t = 1j*np.arange(1/2, q+1/2)*Kp/q

    sn_, cn_, dn_, _ = sps.ellipj(np.imag(t), 1-k2)  # compute real valued functions to transform to what we need

    cn = 1/cn_
    dn = dn_ / cn_
    sn = 1j * sn_ / cn_

    poles = np.real(a * sn**2)
    weights = ((2 * Kp * np.sqrt(a))/(np.pi*q)) * (cn * dn)
    return weights, poles


def inv_sqrt_rat(q, a, b):
    weights, poles = inv_sqrt_weights_poles(q, a, b)
    def approx_inv_sqrt(x): return np.sum(weights / (x[:, None] - poles), axis=1)
    approx_inv_sqrt.degree = (q-1, q)
    return approx_inv_sqrt


def sqrt_rat(q, a, b):
    inv_rat = inv_sqrt_rat(q, a, b)
    def approx_sqrt(x): return x * inv_rat(x)
    approx_sqrt.degree = (q, q)
    return approx_sqrt


class DiagonalSqrtAProblem(DiagonalFAProblem):
    def __init__(self, spectrum, b, cache_k=None):
        super().__init__(flamp.sqrt, spectrum, b, cache_k=cache_k)

    @cache
    def zolotarev_approx(self, q):
        return sqrt_rat(q, self.spectrum.min(), self.spectrum.max())

    def zolotarev_error(self, q):
        return norm(
            self.zolotarev_approx(q)(self.spectrum) * self.b - self.ground_truth()
        )

    def zolotarev_lanczos_error(self, q, k):
        return self.lanczos_on_approximant_error(k, self.zolotarev_approx(q))

    def ciq(self, q, k):
        weights, poles = inv_sqrt_weights_poles(
            q, self.spectrum.min(), self.spectrum.max()
        )
        decomp = self.lanczos_decomp(k)
        inv_sqrt = sum(w * decomp.shift(-p).minres() for w, p in zip(weights, poles))
        return self.A() @ inv_sqrt

    def ciq_error(self, q, k):
        ciq_estimate = self.ciq(q, k)
        error = ciq_estimate - self.ground_truth()
        return norm(error)


# Tests of Zolotarev, compare with Hale
# import matplotlib.pyplot as plt
# a = 1
# b = 100
# xxx = np.linspace(a, b, 1_000_000)
# approx = sqrt_rat(6, a, b)
# plt.plot(xxx, np.sqrt(xxx), xxx, approx(xxx))
# plt.plot(xxx, approx(xxx) - np.sqrt(xxx))
# plt.plot(xxx, (approx(xxx) - np.sqrt(xxx)) / np.sqrt(xxx))
# qs = list(range(1, 10))
# approxs = [sqrt_rat(q, a, b) for q in qs]
# truth = np.sqrt(xxx)
# errors = [np.max(np.abs(approx(xxx) - truth)) for approx in approxs]
# plt.semilogy(qs, errors)

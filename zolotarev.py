import numpy as np
import scipy.special as sps

# Zolotarev found polynomials $p, q$ such that
# $$r(x) = \frac{xp(x^2)}{q(x^2)} \approx \mathrm{sgn}(x)$$
# on $x \in [-1, -\gamma] \cup [\gamma, 1]$ for $0 < \gamma < 1$.
# Letting $s(x) = p(x) / q(x)$, this means
# $$s(x^2) = \frac{r(x)}x \approx \frac{\mathrm{sgn}(x)}{x} = \frac1{|x|}$$
# $$\implies |x| \approx \frac1{s(x^2)}$$
# Letting $y = x^2$, this yields
# $$\sqrt{y} \approx \frac1{s(y)} = \frac{q(y)}{p(y)}$$
# for all $y \in [\gamma^2, 1]$.

# Papers
# https://www.math.ucdavis.edu/~freund/zolopaper.pdf (page 7)
# https://arxiv.org/pdf/1910.06517.pdf


def tyler_inv_sqrt(q, a, b):
    # TODO: should we change uses of np to flamp?
    # TODO: when a,b are extraprecision, see below
    # https://mpmath.org/doc/current/functions/elliptic.html?highlight=elliptic#jacobi-elliptic-functions

    # In the sidford paper, the approximation is valid in the range [gamma, 1]
    # that corresponds to [min(abs(lam)), max(abs(lam))] / max(abs(lam))
    # since sidford's gamma =tyler's sqrt(k2)

    # lmin = np.min(np.abs(lam))**2
    # lmax = np.max(np.abs(lam))**2
    assert 0 < a < b
    lmin = a ** 2
    lmax = b ** 2

    poles = np.zeros(q)
    weights = np.zeros(q)

    k2 = lmin/lmax  # this is called gamma^2 in the Sidford paper
    Kp = sps.ellipk(1-k2)  # this is called K prime in the Sidford paper

    t = 1j*np.arange(1/2, q+1/2)*Kp/q

    sn_, cn_, dn_, _ = sps.ellipj(np.imag(t), 1-k2)  # compute real valued functions to transform to what we need

    cn = 1/cn_
    dn = dn_ / cn_
    sn = 1j * sn_ / cn_

    poles = np.real(lmin * sn**2)

    weights = (2 * Kp * np.sqrt(lmin))/(np.pi*q) * (cn * dn)

    def approx_inv_sqrt(x): return np.sum(weights / (x[:, None] - poles), axis=1)
    # TODO! check this, it matches the code but not what's written in the papers
    # there it seems like it should be (q, q)
    approx_inv_sqrt.degree = (q-1, q)
    return approx_inv_sqrt


def tylers_sign(q, a, b):
    f = tyler_inv_sqrt(q, a, b)
    f_num, f_denom = f.degree

    def approx_sign(x): return x * f(x ** 2)
    approx_sign.degree = (2 * f_num + 1, 2 * f_denom)

    return approx_sign


def tylers_sqrt(q, a, b):
    f = tyler_inv_sqrt(q, a, b)
    f_num, f_denom = f.degree

    def approx_sqrt(x): return 1./f(x)
    approx_sqrt.degree = (f_denom, f_num)
    return approx_sqrt


def my_poly(x):
    return 1 + x * (1/2 + x * (1/9 + x * (1/72 + x * (1/1008 + x / 30240))))


def exp_pade0_55(x):
    return my_poly(x) / my_poly(-x)


exp_pade0_55.degree = (5, 5)

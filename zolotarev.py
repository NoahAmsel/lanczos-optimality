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


def tyler_inv_sqrt(q, a, b):
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

    t = 1j*np.arange(1/2,q+1/2)*Kp/q

    sn_,cn_,dn_,_ = sps.ellipj(np.imag(t),1-k2) # compute real valued functions to transform to what we need

    cn = 1/cn_
    dn = dn_ / cn_
    sn = 1j * sn_ / cn_

    poles = np.real(lmin * sn**2)

    weights = (2 * Kp * np.sqrt(lmin))/(np.pi*q) * (cn * dn)

    return lambda x: np.sum( weights / (x[:,None]-poles) ,axis=1)


def tylers_sign(q, a, b):
    f = tyler_inv_sqrt(q, a, b)
    return lambda x: x * f(x ** 2)


def tylers_sqrt(q, a, b):
    f = tyler_inv_sqrt(q, a, b)
    return lambda x: 1./f(x)

import warnings

import flamp
import matplotlib.pyplot as plt
import numpy as np


def remez(f, k, domain=[-1, 1], max_iter=100, n_grid=2000, tol=1e-10):

    a, b = domain

    def ff(x): return f(x*(b-a)/2 + (a+b)/2).astype(float)  # scale f to [-1,1]

    X = np.cos(np.arange(k+2)*np.pi/(k+1))
    D = np.unique(np.hstack([X, np.cos(np.arange(n_grid+1)*np.pi/n_grid)]))

    for iter in range(max_iter):

        V = np.vstack([np.polynomial.chebyshev.chebvander(
            X, k).T, (-1)**np.arange(k+2)]).T
        fX = ff(X)

        c = np.linalg.solve(V, fX)
        E = np.abs(c[-1])  # lower bound on infinity norm error

        p = np.polynomial.chebyshev.Chebyshev(c[:-1])

        intersections = np.nonzero(np.diff(np.sign(p(D)-ff(D))))[0]
        edges = np.concatenate([[0], intersections, [None]])

        # upper bounbds on infinity norm error
        F = np.max(np.abs(ff(D) - p(D)))

        if np.abs(E-F) < tol:
            return E, F, X*(b-a)/2 + (a+b)/2

        # for i in range(k+2):
        for i in range(min(len(edges)-1, len(X))):

            if edges[i] == edges[i+1]:
                X[i] = D[edges[i]]
            else:
                Di = D[edges[i]:edges[i+1]]
                index_in_interval = np.argmax(np.abs(ff(Di)-p(Di)))
                X[i] = D[edges[i] + index_in_interval]

    warnings.warn(f"Remez fail to converge after {max_iter} iterations")
    return E, F, X*(b-a)/2 + (a+b)/2


def flamp_arange(n):
    return flamp.to_mp(np.arange(n))


def remez_flamp(f, k, domain=[-1, 1], max_iter=100, n_grid=2000, tol=1e-10):

    a, b = domain
    a = flamp.gmpy2.mpfr(a)
    b = flamp.gmpy2.mpfr(b)

    def ff(x): return f(x*(b-a)/2 + (a+b)/2)  # scale f to [-1,1]

    X = flamp.cos(flamp_arange(k+2)*flamp.gmpy2.const_pi()/(k+1))
    D = np.unique(np.hstack([X, flamp.cos(flamp_arange(n_grid+1)*flamp.gmpy2.const_pi()/n_grid)]))

    for iter in range(max_iter):

        V = np.vstack([np.polynomial.chebyshev.chebvander(
            X, k).T, (-1)**flamp_arange(k+2)]).T
        fX = ff(X)

        c = flamp.lu_solve(V, fX)
        E = np.abs(c[-1])  # lower bound on infinity norm error

        p = np.polynomial.chebyshev.Chebyshev(c[:-1])

        intersections = np.nonzero(np.diff(np.sign(p(D)-ff(D))))[0]
        edges = np.concatenate([[0], intersections, [None]])

        # upper bounbds on infinity norm error
        F = np.max(np.abs(ff(D) - p(D)))

        if np.abs(E-F) < tol:
            return E, F, X*(b-a)/2 + (a+b)/2

        # for i in range(k+2):
        for i in range(min(len(edges)-1, len(X))):

            if edges[i] == edges[i+1]:
                X[i] = D[edges[i]]
            else:
                Di = D[edges[i]:edges[i+1]]
                index_in_interval = np.argmax(np.abs(ff(Di)-p(Di)))
                X[i] = D[edges[i] + index_in_interval]

    warnings.warn(f"Remez fail to converge after {max_iter} iterations")
    return E, F, X*(b-a)/2 + (a+b)/2


if __name__ == "__main__":
    problems = [
        {'f': lambda x: x**(-2),
         'domain': [1, 100],
         'tol':1e-5},
        {'f': lambda x: flamp.exp((x-100)),
         'domain': [1, 100],
         'tol':1e-9},
        {'f': lambda x: flamp.sqrt(x),
         'domain': [1, 100],
         'tol':1e-7},
    ]

    k_max = 50

    lowers = []
    uppers = []
    for problem in problems:
        f = problem['f']
        domain = problem['domain']
        tol = problem['tol']

        E = np.zeros(k_max)
        F = np.zeros(k_max)
        for k in range(k_max):

            E[k], F[k], _ = remez(f, k, domain=domain, max_iter=400, tol=tol)

        lowers.append(E)
        uppers.append(F)

    # cheb_errors = []
    # for problem in problems:
    #     f = problem['f']
    #     domain = problem['domain']

    #     E = np.zeros(k_max)
    #     for k in range(k_max):
    #         E[k] = cheb_err(f,k,domain=domain)
    #     cheb_errors.append(E)

    for i, problem in enumerate(problems):

        plt.plot()
        plt.plot(uppers[i])
        plt.plot(lowers[i], ls=':')
        # plt.plot(cheb_errors[i])

        plt.yscale('log')
        plt.show()

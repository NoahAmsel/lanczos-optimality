import warnings

import flamp
import matplotlib.pyplot as plt
import numpy as np


def remez(f, degree, domain=[-1, 1], max_iter=100, n_grid=2000, tol=1e-10):

    a, b = domain

    def ff(x): return f(x*(b-a)/2 + (a+b)/2).astype(float)  # scale f to [-1,1]

    X = np.cos(np.arange(degree+2)*np.pi/(degree+1))
    D = np.unique(np.hstack([X, np.cos(np.arange(n_grid+1)*np.pi/n_grid)]))

    for iter in range(max_iter):

        V = np.vstack([np.polynomial.chebyshev.chebvander(
            X, degree).T, (-1)**np.arange(degree+2)]).T
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


def remez_flamp(f, degree, domain=[-1, 1], max_iter=100, n_grid=2000, tol=1e-10):
    a, b = domain
    a = flamp.gmpy2.mpfr(a)
    b = flamp.gmpy2.mpfr(b)
    D = flamp.cos(flamp_arange(n_grid+1)*flamp.gmpy2.const_pi()/n_grid)
    D = D*(b-a)/2 + (a+b)/2
    return remez_flamp_grid(f, degree, D, max_iter=max_iter, tol=tol)


# def remez_flamp_grid(f, degree, points, max_iter=100, tol=1e-10):
#     points = np.array(points)
#     a = points.min()
#     b = points.max()
#     def shift(x): return 2*(x-a)/(b-a) - 1
#     def ff(x): return f(x*(b-a)/2 + (a+b)/2)  # scale f to [-1,1]

#     X = flamp.cos(flamp_arange(degree+2)*flamp.gmpy2.const_pi()/(degree+1))
#     # DOES IT SUFFICE TO JUST SET D (AND MAYBE INITIALIZE X??) TO BEING THE SPECTRUM?
#     # D = np.unique(np.hstack([X, flamp.cos(flamp_arange(n_grid+1)*flamp.gmpy2.const_pi()/n_grid)]))
#     # D = flamp.cos(flamp_arange(n_grid+1)*flamp.gmpy2.const_pi()/n_grid)
#     points = shift(points)

#     for iter in range(max_iter):

#         V = np.vstack([np.polynomial.chebyshev.chebvander(
#             X, degree).T, (-1)**flamp_arange(degree+2)]).T
#         fX = ff(X)

#         c = flamp.lu_solve(V, fX)
#         E = np.abs(c[-1])  # lower bound on infinity norm error

#         p = np.polynomial.chebyshev.Chebyshev(c[:-1])

#         intersections = np.nonzero(np.diff(np.sign(p(points)-ff(points))))[0]
#         edges = np.concatenate([[0], intersections, [None]])

#         # upper bounbds on infinity norm error
#         F = np.max(np.abs(ff(points) - p(points)))

#         if np.abs(E-F) < tol:
#             return E, F, X*(b-a)/2 + (a+b)/2

#         # # problem is that we initialize with a chebyshev approx, so no reason for the err function to have the right number of roots/extrema
#         # X = find_peaks(err_fun)
#         # assert len(X) == degree + 2

#         # for i in range(k+2):
#         for i in range(min(len(edges)-1, len(X))):

#             if edges[i] == edges[i+1]:
#                 X[i] = points[edges[i]]
#             else:
#                 Di = points[edges[i]:edges[i+1]]
#                 index_in_interval = np.argmax(np.abs(ff(Di)-p(Di)))
#                 X[i] = points[edges[i] + index_in_interval]

#     warnings.warn(f"Remez fail to converge after {max_iter} iterations")
#     print("fails")
#     return E, F, X*(b-a)/2 + (a+b)/2


from scipy.signal import find_peaks
import matrix_functions as mf

def remez_flamp_grid(f, degree, points, max_iter=100, tol=1e-10):
    points = np.array(points)
    a = points.min()
    b = points.max()
    def shift(x): return 2*(x-a)/(b-a) - 1  # send [a,b] to [-1, 1]
    def unshift(x): return x*(b-a)/2 + (a+b)/2  # send [-1, 1] to [a, b]
    def ff(x): return f(unshift(x))
    points = shift(points)
    f_points = ff(points)

    # initial guess: chebyshev l2 regression
    CV = mf.cheb_vandermonde(points, degree)
    my_coeffs = flamp.qr_solve(CV, f_points)
    my_errors = CV @ my_coeffs - f_points
    X = points[np.sort(np.hstack((0, find_peaks(my_errors)[0], find_peaks(-my_errors)[0], len(points)-1)))]

    # TODO: I added this block to handle the special case where the function is interpolated perfectly by Chebyshev regression
    # but I'm not sure it's the right thing
    if (len(X) < degree + 2) and (np.max(np.abs(my_errors)) < tol):
        return 0, np.max(np.abs(my_errors)), X

    # # X = flamp.cos(flamp_arange(degree+2)*flamp.gmpy2.const_pi()/(degree+1))
    # spacing = len(points) // (degree+3)
    # X = points[list(range(spacing, spacing*(degree+3), spacing))]
    # # DOES IT SUFFICE TO JUST SET D (AND MAYBE INITIALIZE X??) TO BEING THE SPECTRUM?
    # # D = np.unique(np.hstack([X, flamp.cos(flamp_arange(n_grid+1)*flamp.gmpy2.const_pi()/n_grid)]))
    # # D = flamp.cos(flamp_arange(n_grid+1)*flamp.gmpy2.const_pi()/n_grid)

    for iter in range(max_iter):
        V = np.hstack((
            mf.cheb_vandermonde(X, degree), ((-1)**flamp_arange(degree+2))[:, np.newaxis]
        ))

        # V = np.vstack([np.polynomial.chebyshev.chebvander(
        #     X, degree).T, (-1)**flamp_arange(degree+2)]).T
        fX = ff(X)

        c = flamp.lu_solve(V, fX)
        E = np.abs(c[-1])  # lower bound on infinity norm error

        p = np.polynomial.chebyshev.Chebyshev(c[:-1])

        # upper bounbds on infinity norm error
        err_fun = f_points - p(points)
        F = np.max(np.abs(err_fun))

        if np.abs(E-F) < tol:
            return E, F, X*(b-a)/2 + (a+b)/2

        # problem is that we initialize with a chebyshev approx, so no reason for the err function to have the right number of roots/extrema
        # extrema_indices = np.hstack((find_peaks(err_fun)[0], find_peaks(-err_fun)[0]))
        # if (len(extrema_indices) < degree + 2) and (extrema_indices[0] != 0):
        #     extrema_indices = np.hstack((0, extrema_indices))
        # if (len(extrema_indices) < degree + 2) and (extrema_indices[-1] != len(err_fun)-1):
        #     extrema_indices = np.hstack((extrema_indices, len(err_fun)-1))
        extrema_indices = np.sort(np.hstack((0, find_peaks(err_fun)[0], find_peaks(-err_fun)[0], len(err_fun)-1)))
        assert len(extrema_indices) == degree + 2
        X = points[extrema_indices]

    warnings.warn(f"Remez fail to converge after {max_iter} iterations")
    print("fails")
    return E, F, X*(b-a)/2 + (a+b)/2


from matrix_functions import cheb_interpolation
from matrix_functions.utils import linspace


def cheb_err(f, degree, domain):
    interp = cheb_interpolation(degree=degree, f=f, a=domain[0], b=domain[1], dtype=np.dtype('O'))
    grid = linspace(domain[0], domain[1], num=1_000, endpoint=True, dtype=np.dtype('O'))
    return np.abs(f(grid) - interp(grid)).max()


if __name__ == "__main__":
    flamp.set_dps(100)
    def p(x): return x**3 - 2*(x**2) + 6*x - 24
    spectrum = linspace(3, 4, num=9, dtype=np.dtype('O'))
    e, f, _ = remez_flamp_grid(f=p, degree=3, points=spectrum, max_iter=100, tol=1e-16)
    print(e)
    print(f)
    print(cheb_err(f=p, degree=3, domain=[3, 4]))

if __name__ == "__main__":
    flamp.set_dps(100)
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

    k_max = 30

    lowers = []
    uppers = []
    all_ks = []
    chebs = []
    for problem in problems:
        f = problem['f']
        domain = problem['domain']
        tol = problem['tol']

        k_list = []
        E = []
        F = []
        C = []
        for k in range(1, k_max+1, 10):
            this_e, this_f, _ = remez_flamp(f, k, domain=domain, max_iter=400, tol=tol)
            k_list.append(k)
            E.append(this_e)
            F.append(this_f)
            C.append(cheb_err(f,k,domain=domain))

        all_ks.append(k_list)
        lowers.append(E)
        uppers.append(F)
        chebs.append(C)



    for i, problem in enumerate(problems):

        plt.plot()
        plt.plot(all_ks[i], uppers[i])
        plt.plot(all_ks[i], lowers[i], ls=':')
        plt.plot(all_ks[i], chebs[i])

        plt.yscale('log')
        plt.show()

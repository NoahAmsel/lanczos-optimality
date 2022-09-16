import numpy as np
import numpy.linalg as lin
import scipy.linalg

def generate_orthonormal(n, seed=None):
    M = np.random.default_rng(seed).normal(size=(n, n))
    Q, _ = np.linalg.qr(M)
    return Q

def generate_symmetric(eigenvalues, seed=None):
    Q = generate_orthonormal(len(eigenvalues), seed=seed)
    return Q @ np.diag(eigenvalues) @ Q.T

def naive_fa(f, A, x):
    l, V = lin.eigh(A)
    # A = V @ np.diag(l) @ V.T
    return V @ (f(l) * (V.T @ x))

def diagonal_fa(f, a_diag, x):
    return f(a_diag) * x

def lanczos_fa(f, A, x, k=None):
    """See Stability of Lanczos Method page 5
    """
    assert np.allclose(A, A.T)  # this probably hurts performance
    n = A.shape[0]

    if k is None:
        k = n

    Q = [x / lin.norm(x)]
    alpha = []
    beta = []  # this is really beta_2, beta_3, ...
    for i in range(k):
        if i == 0:
            q_i_minus_1 = np.zeros(n)
            beta_i = 0
        else:
            q_i_minus_1 = Q[-2]
            beta_i = beta[-1]

        q_i = Q[-1]

        next_q = A @ q_i - beta_i * q_i_minus_1
        alpha_i = np.inner(next_q, q_i)
        alpha.append(alpha_i)
        next_q -= alpha_i * q_i
        next_beta = lin.norm(next_q)

        if (i == k-1) or (next_beta == 0):  # TODO: numerical zero?
            break

        next_q /= next_beta
        Q.append(next_q)
        beta.append(next_beta)

    Q = np.column_stack(Q)
    alpha = np.array(alpha)
    beta = np.array(beta)

    # We don't need to construct T, but if we did it would be this:
    # T = scipy.sparse.diags([beta, alpha, beta], [-1, 0, 1]) # .toarray()

    T_lambda, T_V = scipy.linalg.eigh_tridiagonal(alpha, beta)    
    return lin.norm(x) * (Q @ (T_V @ (f(T_lambda) * T_V[0, :])))

if __name__ == "__main__":
    pass

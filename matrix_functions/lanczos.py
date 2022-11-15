import numpy as np

from .utils import norm


def lanczos(A, q_1, k=None, reorthogonalize=False, beta_tol=0):
    """See Stability of Lanczos Method page 5
    """
    # assert np.allclose(A, A.T)  # can't do this with sparse matrix
    n = len(q_1)
    assert A.shape[0] == A.shape[1] == n

    if k is None:
        k = n

    result_type = np.result_type(A, q_1)
    if np.issubdtype(result_type, np.integer):
        result_type = np.float64
    Q = np.empty((n, k), dtype=result_type)
    alpha = np.empty(k, dtype=result_type)
    beta = np.empty(k-1, dtype=result_type)  # this is really beta_2, beta_3, ...

    Q[:, 0] = q_1 / norm(q_1)
    next_q = A @ Q[:, 0]
    alpha[0] = np.inner(next_q, Q[:, 0])
    next_q -= alpha[0] * Q[:, 0]

    for i in range(1, k):
        beta[i-1] = norm(next_q)
        if beta[i-1] <= beta_tol:
            # TODO: in some cases we may just want to continue from a new vector
            return np.atleast_2d(Q[:, :i]), (alpha[:i], beta[:(i-1)])

        Q[:, i] = next_q / beta[i-1]

        next_q = A @ Q[:, i]
        alpha[i] = np.inner(next_q, Q[:, i])
        next_q -= alpha[i] * Q[:, i]
        next_q -= beta[i-1] * Q[:, i-1]

        if reorthogonalize:
            for _ in range(2):
                next_q -= Q[:, :(i+1)] @ (Q[:, :(i+1)].T @ next_q)

    return Q, (alpha, beta)

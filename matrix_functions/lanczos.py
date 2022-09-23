import numpy as np
import numpy.linalg as lin


def lanczos(A, q_1, k=None, reorthogonalize=False):
    """See Stability of Lanczos Method page 5
    """
    # assert np.allclose(A, A.T)  # can't do this with sparse matrix
    n = len(q_1)
    assert A.shape[0] == A.shape[1] == n

    if k is None:
        k = n

    Q = np.empty((n, k))
    alpha = np.empty(k)
    beta = np.empty(k-1)  # this is really beta_2, beta_3, ...

    Q[:, 0] = q_1 / lin.norm(q_1)
    next_q = A @ Q[:, 0]
    alpha[0] = np.inner(next_q, Q[:, 0])
    next_q -= alpha[0] * Q[:, 0]

    for i in range(1, k):
        beta[i-1] = lin.norm(next_q)
        # TODO: what if beta is 0
        Q[:, i] = next_q / beta[i-1]

        next_q = A @ Q[:, i]
        alpha[i] = np.inner(next_q, Q[:, i])
        next_q -= alpha[i] * Q[:, i]
        next_q -= beta[i-1] * Q[:, i-1]

        if reorthogonalize:
            for _ in range(2):
                next_q -= Q[:, :(i+1)] @ (Q[:, :(i+1)].T @ next_q)

    return Q, (alpha, beta)

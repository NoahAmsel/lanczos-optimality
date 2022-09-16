import numpy as np
import numpy.linalg as lin

def lanczos(A, q_1, k=None):
    """See Stability of Lanczos Method page 5
    """
    assert np.allclose(A, A.T)  # this probably hurts performance
    n = A.shape[0]

    if k is None:
        k = n

    Q = [q_1 / lin.norm(q_1)]
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

    return Q, alpha, beta

import numpy as np
import numpy.linalg as lin

def generate_orthonormal(n, seed=None):
    M = np.random.default_rng(seed).normal(size=(n, n))
    Q, _ = lin.qr(M)
    return Q

def generate_symmetric(eigenvalues, seed=None):
    Q = generate_orthonormal(len(eigenvalues), seed=seed)
    return Q @ np.diag(eigenvalues) @ Q.T

def krylov_subspace(A, x, size=None):
    """Do Gram-Schmidt orthogonalization along the way to avoid ill-conditioning
    """
    if size is None:
        size = len(A)

    basis = np.empty((len(x), size))
    basis[:, 0] = x / lin.norm(x)
    for i in range(1, size):
        basis[:, i] = A @ basis[:, i-1]
        # In exact arithmetic you could just do this, but it's numerically unstable
        # basis[:, i] -= basis[:, :i] @ (basis[:, :i].T @ basis[:, i])
        # Instead, project one by one so that round off errors don't get amplified
        for j in range(i):
            basis[:, i] -= (basis[:, i] @ basis[:, j]) * basis[:, j]
        basis[:, i] /= lin.norm(basis[:, i])
    return basis

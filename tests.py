import numpy as np
import numpy.linalg as lin

from function_approx import *
from lanczos import lanczos

def generate_orthonormal(n, seed=None):
    M = np.random.default_rng(seed).normal(size=(n, n))
    Q, _ = lin.qr(M)
    return Q

def generate_symmetric(eigenvalues, seed=None):
    Q = generate_orthonormal(len(eigenvalues), seed=seed)
    return Q @ np.diag(eigenvalues) @ Q.T

if __name__ == "__main__":
    a_diag = np.random.randn(3)
    A = np.diag(a_diag)    
    b = np.random.rand(3)

    ref = b / a_diag
    assert np.allclose(naive_fa(np.reciprocal, A, b), ref)
    assert np.allclose(diagonal_fa(np.reciprocal, a_diag, b), ref)
    assert np.allclose(lanczos_fa(np.reciprocal, A, b), ref)

if __name__ == "__main__":
    A = np.random.randn(3, 3)
    A = A+A.T  # A should be symmetric
    b = np.random.rand(3)
    ref = lin.solve(A, b)
    assert np.allclose(naive_fa(np.reciprocal, A, b), ref)
    assert np.allclose(lanczos_fa(np.reciprocal, A, b), ref)

if __name__ == "__main__":
    for dim in [4, 1000]:
        A = generate_symmetric(list(range(dim//2)) + [0]*(dim//2))
        b = np.random.rand(dim)
        ref = naive_fa(np.exp, A, b)
        assert np.allclose(lanczos_fa(np.exp, A, b, k=dim//2), ref)

if __name__ == "__main__":
    dim = 10
    A = generate_symmetric(list(range(dim//2)) + [0]*(dim//2))
    b = np.random.rand(dim)
    for ks in [range(1, 10), [10, 1, 4]]:
        for k, truncated_estimate in zip(ks, lanczos_fa_multi_k(np.exp, A, b, ks=ks)):
            assert np.allclose(truncated_estimate, lanczos_fa(np.exp, A, b, k=k))
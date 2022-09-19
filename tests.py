import numpy as np
import numpy.linalg as lin

from function_approx import *
from lanczos import lanczos
from utils import *

if __name__ == "__main__":
    a_diag = np.random.rand(3)
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

if __name__ == "__main__":
    A = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    x = np.ones(3)
    ref = np.column_stack([
        np.ones(3)/np.sqrt(3), 
        np.array([1, 0, -1])/np.sqrt(2),
        np.array([-np.sqrt(2), 2*np.sqrt(2), -np.sqrt(2)])/np.sqrt(12)
    ])
    for k in range(1, 4):
        assert np.allclose(krylov_subspace(A, x, k), ref[:, :k])

if __name__ == "__main__":
    dim = 100
    A = np.random.randn(dim, dim)
    x = np.random.randn(dim)
    Q = krylov_subspace(A, x)
    assert np.allclose(Q.T @ Q, np.eye(dim))

# if __name__ == "__main__":
#     dim = 100
#     a_diag = np.array(list(range(1, dim+1)))
#     A = np.diag(a_diag)
#     x = np.ones(dim)
#     Q = krylov_subspace(A, x)
#     assert np.allclose(Q.T @ Q, np.eye(dim), rtol=1e-4, atol=1e-4)

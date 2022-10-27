import numpy as np
import numpy.linalg as lin
from scipy import sparse

import matrix_functions as mf


def test_fa_diagonal():
    a_diag = np.random.rand(3)
    A = np.diag(a_diag)
    A_sparse = sparse.diags((a_diag), (0))
    b = np.random.rand(3)

    ref = b / a_diag
    assert np.allclose(mf.naive_fa(np.reciprocal, A, b), ref)
    assert np.allclose(mf.diagonal_fa(np.reciprocal, a_diag, b), ref)
    assert np.allclose(mf.lanczos_fa(np.reciprocal, A_sparse, b), ref)


def test_fa_linear_system():
    A = np.random.randn(3, 3)
    A = A+A.T  # A should be symmetric
    b = np.random.rand(3)
    ref = lin.solve(A, b)
    assert np.allclose(mf.naive_fa(np.reciprocal, A, b), ref)
    assert np.allclose(mf.lanczos_fa(np.reciprocal, A, b), ref)


def test_lanczos_fa_exponential():
    for dim in [4, 1000]:
        A = mf.generate_symmetric(list(range(dim//2)) + [0]*(dim//2))
        b = np.random.rand(dim)
        ref = mf.naive_fa(np.exp, A, b)
        assert np.allclose(mf.lanczos_fa(np.exp, A, b, k=dim//2), ref)


def test_lanczos_fa_multi_k():
    dim = 10
    A = mf.generate_symmetric(list(range(dim//2)) + [0]*(dim//2))
    b = np.random.rand(dim)
    for ks in [range(1, 10), [10, 1, 4]]:
        for k, truncated_estimate in zip(ks, mf.lanczos_fa_multi_k(np.exp, A, b, ks=ks)):
            assert np.allclose(truncated_estimate, mf.lanczos_fa(np.exp, A, b, k=k))

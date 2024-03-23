import numpy as np

import matrix_functions as mf


def test_minres():
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    A = B.T @ B
    x = np.array([1, 2, 3])
    b = A @ x
    decomp = mf.LanczosDecomposition.fit(A, b)
    assert np.allclose(decomp.cg(), x)
    assert np.allclose(decomp.minres(), x)

    z = 10
    shifted_decomp = decomp.shift(z)
    assert np.allclose((A + z * np.eye(3)) @ shifted_decomp.cg(), b)
    assert np.allclose((A + z * np.eye(3)) @ shifted_decomp.minres(), b)

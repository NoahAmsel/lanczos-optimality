import numpy as np

import matrix_functions as mf


def test_start_vec():
    dim = 500
    eigs = np.arange(dim) * 2 + 1
    desired_ritz = eigs[:-1] + 1
    p1 = mf.start_vec(eigs, desired_ritz)
    _, (a, b), _ = mf.lanczos(mf.DiagonalMatrix(eigs), p1, dim - 1, reorthogonalize=False)
    true_ritz, _ = mf.utils.eigh_tridiagonal(a, b)
    assert np.allclose(true_ritz, desired_ritz)

import numpy as np

import matrix_functions as mf


def test_DiagonalMatrix():
    A = mf.DiagonalMatrix(np.array([1, 2, 3], dtype=np.int32))
    assert len(A) == 3
    assert A.dtype == np.int32
    assert A.shape == (3, 3)
    b = np.array([-1, 0, 1])
    assert np.all(A @ b == np.array([-1, 0, 3]))

    C = np.vstack([b, np.ones(3)]).T
    assert np.all(A @ C == np.array([[-1, 1], [0, 2], [3, 3]]))

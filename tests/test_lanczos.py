import numpy as np

from matrix_functions import *

def test_lanczos_exactly():
    A = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    x = np.ones(3)
    ref = np.column_stack([
        np.ones(3)/np.sqrt(3), 
        np.array([1, 0, -1])/np.sqrt(2),
        np.array([-np.sqrt(2), 2*np.sqrt(2), -np.sqrt(2)])/np.sqrt(12)
    ])
    for k in range(1, 4):
        assert np.allclose(lanczos(A, x, k)[0], ref[:, :k])

def test_krylov_orthonormality_diagonal():
    dim = 500
    a_diag = np.array(list(range(1, dim+1)))
    A = np.diag(a_diag)
    x = np.ones(dim)
    Q, _ = lanczos(A, x, reorthogonalize=True)
    assert np.allclose(Q.T @ Q, np.eye(dim), rtol=1e-4, atol=1e-4)

def test_krylov_orthonormality():
    dim = 100
    A = generate_model_problem(dim, 100, 0.5)
    x = np.random.randn(dim)
    Q, _ = lanczos(A, x, reorthogonalize=True)
    assert np.allclose(Q.T @ Q, np.eye(dim))

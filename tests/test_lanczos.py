import flamp
import numpy as np

import matrix_functions as mf


def test_lanczos_exactly():
    A = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    x = np.ones(3)
    ref = np.column_stack(
        [
            np.ones(3) / np.sqrt(3),
            np.array([1, 0, -1]) / np.sqrt(2),
            np.array([-np.sqrt(2), 2 * np.sqrt(2), -np.sqrt(2)]) / np.sqrt(12),
        ]
    )
    for k in range(1, 4):
        assert np.allclose(mf.lanczos(A, x, k)[0], ref[:, :k])


def test_lanczos_early_stop():
    A = np.diag([1.0, 2, 3, 4])
    x = np.array([1, 0, 1, 0])
    Q, (alpha, beta), next_q = mf.lanczos(A, x, beta_tol=1e-14)
    assert Q.shape == (4, 2)
    assert np.allclose(
        Q @ mf.SymmetricTridiagonal(alpha, beta).to_sparse() @ Q.transpose() @ x, A @ x
    )
    assert np.allclose(next_q, np.zeros(4))


def test_krylov_orthonormality_diagonal():
    dim = 500
    a_diag = np.array(list(range(1, dim + 1)))
    A = np.diag(a_diag)
    x = np.ones(dim)
    Q, _, next_q = mf.lanczos(A, x, reorthogonalize=True)
    assert np.allclose(Q.T @ Q, np.eye(dim), rtol=1e-4, atol=1e-4)
    assert np.allclose(next_q, np.zeros(dim))


def test_krylov_orthonormality():
    dim = 100
    A = mf.generate_symmetric(mf.model_spectrum(dim, 100, 0.5))
    x = np.random.randn(dim)
    Q, _, _ = mf.lanczos(A, x, reorthogonalize=True)
    assert np.allclose(Q.T @ Q, np.eye(dim))


def test_high_precision():
    with flamp.extraprec(flamp.dps_to_prec(50) - flamp.get_precision()):
        dim = 100
        temp = np.random.standard_normal((dim, dim))
        X = flamp.to_mp(temp + temp.T)

        Q, (alpha, beta), _ = mf.lanczos(X, flamp.ones(dim), reorthogonalize=True)
        T = flamp.zeros((dim, dim))
        np.fill_diagonal(T, alpha)
        np.fill_diagonal(T[:, 1:], beta)
        np.fill_diagonal(T[1:, :], beta)

        X_lanczos = Q @ T @ Q.T

        assert np.linalg.norm(X - X_lanczos, ord=np.inf) < flamp.gmpy2.mpfr("1e-47")
        assert np.linalg.norm(Q @ Q.T - flamp.eye(dim), ord=np.inf) < flamp.gmpy2.mpfr(
            "1e-47"
        )

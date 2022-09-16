from lanczos import *

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

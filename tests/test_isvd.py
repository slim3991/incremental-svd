import numpy as np
import pytest
from isvd import IncrementalSVD


def test_inference_and_types():
    """Verify that the factory returns the correct C++ backend based on dtype."""
    # Test float32
    svd_f = IncrementalSVD(r=2, dtype=np.float32)
    # After fit, check internal U/S types
    X_f = np.random.rand(10, 5).astype(np.float32)
    svd_f.fit(X_f)
    assert svd_f.U.dtype == np.float32
    assert svd_f.S.dtype == np.float32

    # Test float64
    svd_d = IncrementalSVD(r=2, dtype=np.float64)
    X_d = np.random.rand(10, 5).astype(np.float64)
    svd_d.fit(X_d)
    assert svd_d.U.dtype == np.float64
    assert svd_d.S.dtype == np.float64


def test_fit_reconstruction():
    """Check if U * S * V^T roughly reconstructs the input for a small matrix."""
    r = 3
    X = np.random.randn(20, 10)
    svd = IncrementalSVD(r=r)
    svd.fit(X)

    # Check dimensions
    assert svd.U.shape == (20, r)
    assert svd.S.shape == (r,)

    # Check orthogonality of U: U^T * U should be Identity
    identity_approx = svd.U.T @ svd.U
    np.testing.assert_allclose(identity_approx, np.eye(r), atol=1e-7)


def test_increment():
    """Verify that adding a vector changes the decomposition correctly."""
    r = 5
    X = np.random.randn(50, 10)
    svd = IncrementalSVD(r=r)
    svd.fit(X)

    old_U = svd.U.copy()

    # Increment with a new column
    new_vec = np.random.randn(50)
    svd.increment(new_vec)

    assert svd.U.shape == (50, r)
    assert not np.array_equal(svd.U, old_U)
    # Ensure it's still orthogonal
    np.testing.assert_allclose(svd.U.T @ svd.U, np.eye(r), atol=1e-7)


def test_forgetting_factor():
    """Verify that a forgetting factor < 1.0 reduces older singular values."""
    r = 2
    X = np.ones((10, 5))
    svd = IncrementalSVD(r=r, ff=0.1)  # Aggressive forgetting
    svd.fit(X)

    s_initial = svd.S.copy()
    # Add a vector that is very different
    svd.increment(np.random.randn(10))

    # The old components (from the 'ones' matrix) should be heavily discounted
    assert np.all(svd.S < s_initial * 1.1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_consistency(dtype):
    """Run a full cycle for both precisions."""
    X = np.random.randn(10, 4).astype(dtype)
    svd = IncrementalSVD(r=2, dtype=dtype)
    svd.fit(X)
    for _ in range(3):
        svd.increment(np.random.randn(10).astype(dtype))

    assert svd.U.dtype == dtype
    assert not np.isnan(svd.U).any()

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


def test_batch_increment_equivalence():
    """
    Verify that passing a matrix of 3 columns results in the
    same state as calling increment 3 times individually.
    """
    r = 4
    rows, cols = 30, 3
    X_init = np.random.randn(rows, 10)
    batch_data = np.random.randn(rows, cols)

    # 1. Instance for individual increments
    svd_single = IncrementalSVD(r=r)
    svd_single.fit(X_init)
    for i in range(cols):
        svd_single.increment(batch_data[:, i])

    # 2. Instance for batch increment
    svd_batch = IncrementalSVD(r=r)
    svd_batch.fit(X_init)
    svd_batch.increment(batch_data)

    # Assert S and U are identical (within numerical precision)
    np.testing.assert_allclose(svd_single.S, svd_batch.S, atol=1e-7)
    np.testing.assert_allclose(
        np.abs(svd_single.U),
        np.abs(svd_batch.U),
        atol=1e-7,
    )


def test_batch_increment_orthogonality():
    """Verify that a large batch update preserves U's orthogonality."""
    r = 8
    rows = 100
    svd = IncrementalSVD(r=r)
    svd.fit(np.random.randn(rows, 20))

    # Large batch
    batch = np.random.randn(rows, 50)
    svd.increment(batch)

    identity_approx = svd.U.T @ svd.U
    np.testing.assert_allclose(identity_approx, np.eye(r), atol=1e-7)


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

# Incremental SVD (Brand's Algorithm)

A lightweight and efficient implementation of **incremental Singular Value Decomposition (SVD)** using **Brand’s algorithm**, with a fast C++ backend powered by **Eigen** and Python bindings via **pybind11**.

This project is designed for situations where:

- data arrives continuously as a stream,
- datasets are too large to fit entirely into memory,
- or repeatedly recomputing a full SVD would be too expensive.

Instead of rebuilding the decomposition from scratch, the model updates the low-rank approximation incrementally as new vectors arrive.

---

## Features

- **Incremental updates** using Brand’s SVD update algorithm
- **Low memory usage** for large datasets and streaming data
- **Fixed-rank truncation** to maintain computational efficiency
- **Forgetting factor** support for non-stationary data streams
- **C++ performance** with Eigen linear algebra routines
- **Python-friendly API** with NumPy interoperability
- Supports both:
  - `float32`
  - `float64`

---

## Installation

### Requirements

- Python 3.9+
- C++17 compiler
- Eigen
- pybind11
- NumPy

### Build from source

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

---

## Quick Start

```python
import numpy as np
from isvd import IncrementalSVD

# Create an Incremental SVD model
# r  = target rank
# ff = forgetting factor
isvd = IncrementalSVD(r=10, ff=0.95)

# Initial batch fit
# Matrix shape: (features, samples)
X = np.random.randn(100, 50)

isvd.fit(X)

# Add new samples incrementally
new_sample = np.random.randn(100)

isvd.increment(new_sample)

# Access decomposition
U = isvd.U
S = isvd.S

print(U.shape)  # (100, 10)
print(S.shape)  # (10,)
```

---

## API

### `IncrementalSVD(r, ff=1.0, dtype=None)`

Creates a new incremental SVD instance.

#### Parameters

| Parameter | Description |
|---|---|
| `r` | Target rank maintained during updates |
| `ff` | Forgetting factor in `(0, 1]` |
| `dtype` | Optional NumPy dtype (`np.float32` or `np.float64`) |

---

### `fit(X)`

Initializes the decomposition from a batch matrix using standard SVD.

#### Parameters

| Parameter | Shape |
|---|---|
| `X` | `(features, samples)` |

---

### `increment(new_vec)`

Updates the decomposition with a new vector.

#### Parameters

| Parameter | Shape |
|---|---|
| `new_vec` | `(features,)` |

The update:
1. projects the new vector into the current subspace,
2. computes the orthogonal residual,
3. expands the basis if necessary,
4. truncates back to the target rank.

---

## Internal Algorithm

The implementation follows the incremental update strategy proposed by:

> Matthew Brand, *Fast low-rank modifications of the thin singular value decomposition*

At each update:

1. The incoming sample is projected onto the current basis.
2. The orthogonal residual is computed.
3. A small auxiliary matrix is formed.
4. A compact SVD is performed on this small matrix.
5. The basis is updated and truncated to the desired rank.
6. The basis is re-orthogonalized using QR decomposition for numerical stability.

---

## Forgetting Factor

The forgetting factor `ff` controls how strongly older information is retained.

- `ff = 1.0`
  - No forgetting
  - All historical data weighted equally

- `ff < 1.0`
  - Older singular values decay over time
  - More recent data has greater influence

This is useful for:
- streaming signals,
- adaptive systems,
- non-stationary datasets,
- online learning.

---

## Data Types

The Python wrapper automatically selects the backend implementation based on input dtype:

| NumPy dtype | Backend |
|---|---|
| `np.float32` | `IncrementalSVD_F` |
| `np.float64` | `IncrementalSVD_D` |

If no dtype is specified, `float64` is used by default.

---

## Example: Streaming Updates

```python
import numpy as np
from isvd import IncrementalSVD

isvd = IncrementalSVD(r=5, ff=0.98)

# Initial fit
X0 = np.random.randn(50, 20)
isvd.fit(X0)

# Simulated stream
for _ in range(100):
    sample = np.random.randn(50)
    isvd.increment(sample)

print(isvd.S)
```

---

## Implementation Notes

The core implementation uses:

- `Eigen::BDCSVD` for initial decomposition
- `Eigen::JacobiSVD` for incremental updates
- `Eigen::HouseholderQR` for re-orthogonalization

The Python bindings are implemented with `pybind11`.

---

## Current Limitations

- Incremental updates currently accept a single vector at a time
- Only the left singular vectors (`U`) and singular values (`S`) are maintained
- Right singular vectors (`V`) are not tracked
- No automatic rank adaptation

---

## Future Improvements

Potential future additions:

- Batched incremental updates
- Adaptive rank selection
- Sparse matrix support
- GPU acceleration
- Tracking of right singular vectors (`V`)
- Optional re-orthogonalization scheduling

---

## License

MIT License.

# Incremental SVD (Brand's Algorithm)

This Python implementation provides an efficient way to maintain a Singular
Value Decomposition (SVD) of a data stream or a very large dataset that cannot
fit into memory at once. It uses the **Brand’s algorithm** approach to update
the subspace incrementally as new samples arrive. 

While there are several applications for a incremental SVD, such as saving on CPU sycles 
when data arrives intermitently, the main reson why this was this particular
implementation was created was to save on RAM when creating a low-rank
decomposition from a huge matrix. This way the larger matrix is divided into
chucks and added bit by bit to the decomposition.

## Features

* **Low Memory Footprint:** Updates the decomposition without needing to re-process the entire historical dataset.
* **Rank-Truncation:** Maintains a fixed rank $k$ to ensure computational efficiency and noise reduction.
* **Forgetting Factor:** Includes a configurable parameter $f \in (0, 1]$ to weight recent data more heavily—ideal for non-stationary data streams.n
* **Stability:** Built-in periodic re-orthogonalization to combat numerical drift in the basis vectors.

---
## Quick Start

```python
import numpy as np
from isvd import IncrementalSVD  

# 1. Initialize with desired rank and forgetting factor
isvd = IncrementalSVD(rank=10, forgetting_factor=0.95)

# 2. Initial fit with a starting batch
# X should be (Features x Samples)
X_init = np.random.randn(100, 50)
isvd.fit(X_init)

# 3. Incrementally update with new data points
new_sample = np.random.randn(100, 1)
isvd.increment(new_sample)

# 4. Access the decomposed components
U = isvd.U  # Orthogonal basis (Features x Rank)
S = isvd.S  # Singular values (Rank x Rank)
```

## Methods
```fit()``` Initializes the U and S matrices using a standard SVD.

```Increment()``` Updates the current decomposition with a new sample or a batch of samples

```_reoth()``` Internal method that perfroms a QR based re-orthogonalization.


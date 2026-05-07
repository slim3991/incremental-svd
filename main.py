import numpy as np
from isvd import IncrementalSVD


def main():
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


if __name__ == "__main__":
    main()

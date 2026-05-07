from typing import Sequence
import numpy as np
import numpy.typing as npt


class IncrementalSVD:
    def __init__(self, rank: int, forgetting_factor: float = 1.0) -> None:
        self.rank = rank
        self.f = forgetting_factor
        self._is_fitted = False
        self._time_since_reorth = 0
        self.U: npt.NDArray
        self.S: npt.NDArray

    def fit(self, X: npt.NDArray) -> None:
        """Does the first fit, from which the solution is then incremented. If
        the method is called on an already fitted model, the decomposition is reset.
        """
        # Assumes X is (Features x Samples)
        U, S, _ = np.linalg.svd(X, full_matrices=False, compute_uv=True)
        # Handle cases where initial data rank < self.rank
        current_rank = min(self.rank, S.shape[0])
        self.U = U[:, :current_rank]
        self.S = np.diag(S[:current_rank])
        self._is_fitted = True

    def _reoth(self):
        self.U, _ = np.linalg.qr(self.U, mode="reduced")

    def increment(self, new: npt.NDArray) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call fit before increment")

        if new.ndim == 1:
            new = new[:, None]

        # 1. Project and find residual
        m = self.U.T @ new
        p = new - self.U @ m
        p_norm = np.linalg.norm(p)

        # 2. Construct K (the core of the update)
        if p_norm > 1e-10:
            q = p / p_norm
            # K is (k+1, k+1)
            k_top = np.column_stack((np.diag(self.S * self.f), m))
            k_bot = np.append(np.zeros(len(self.S)), p_norm)
            k = np.vstack((k_top, k_bot))

            u_hat, s_hat, _ = np.linalg.svd(k, full_matrices=False)

            # Update U and S
            self.U = np.column_stack((self.U, q)) @ u_hat[:, : self.rank]
            self.S = s_hat[: self.rank]
        else:
            # Subspace didn't expand
            k = np.column_stack((np.diag(self.S * self.f), m))
            u_hat, s_hat, _ = np.linalg.svd(k, full_matrices=False)

            self.U = self.U @ u_hat[:, : self.rank]
            self.S = s_hat[: self.rank]

        # 3. Orthogonality maintenance
        # TODO:: do something more robust than just every 10.
        self._time_since_reorth += 1
        if self._time_since_reorth > 10:
            self._reoth()
            self._time_since_reorth = 0

    def __repr__(self):
        if not self._is_fitted:
            return f"IncrementalSVD(rank={self.rank}, fitted=False)"
        return (
            f"IncrementalSVD(U: {self.U.shape}, S: {self.S.shape}, rank: {self.rank})"
        )

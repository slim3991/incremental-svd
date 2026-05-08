import numpy as np

try:
    # Look for the compiled binary in the current directory
    from isvd.incremental_svd_lib import IncrementalSVD_F, IncrementalSVD_D
except ImportError:
    # This helps debug where Python is actually looking
    import os

    print(
        f"Contents of {os.path.dirname(__file__)}: {os.listdir(os.path.dirname(__file__))}"
    )
    raise


class IncrementalSVD:
    def __init__(self, r, ff=1.0, dtype=None):
        self.r = r
        self.ff = ff
        self._dtype = dtype
        self._instance = None

    def _init_instance(self, data):
        # Infer dtype from data if not explicitly set
        target_dtype = self._dtype if self._dtype else data.dtype

        if target_dtype == np.float32:
            self._instance = IncrementalSVD_F(self.r, self.ff)
        else:
            # Default to double (float64)
            self._instance = IncrementalSVD_D(self.r, self.ff)

    def fit(self, X):
        if self._instance is None:
            self._init_instance(X)
        return self._instance.fit(X)

    def increment(self, new_vec):
        if self._instance is None:
            self._init_instance(new_vec)
        return self._instance.increment(new_vec)

    @property
    def U(self):
        return self._instance.U if self._instance else None

    @property
    def S(self):
        return self._instance.S if self._instance else None

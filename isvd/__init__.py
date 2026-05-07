# This helps Python find the .so if it's sitting in the same folder
try:
    from .incremental_svd_lib import IncrementalSVD
except ImportError:
    # Fallback for different build layouts
    from incremental_svd_lib import IncrementalSVD


class SVDWrapper(IncrementalSVD):
    def __init__(self, r, ff=1.0):
        super().__init__(r, ff)

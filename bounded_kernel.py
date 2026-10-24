import ctypes
from pathlib import Path

import numpy as np


class BoundedKernel:
    def __init__(self, cloud, labels):
        path = Path(__file__).resolve().parent / 'kernels/libgrid.so'
        self.lib = ctypes.CDLL(str(path))
        double_pointer = ctypes.POINTER(ctypes.c_double)
        self.lib.grid_build.restype = ctypes.c_void_p
        self.lib.grid_build.argtypes = [double_pointer, ctypes.POINTER(ctypes.c_int32),
                                        ctypes.c_int64, double_pointer, double_pointer, ctypes.c_double]
        self.lib.grid_query.restype = None
        self.lib.grid_query.argtypes = [ctypes.c_void_p, double_pointer, ctypes.c_int64,
                                        ctypes.POINTER(ctypes.c_uint8), double_pointer]
        self.lib.grid_free.argtypes = [ctypes.c_void_p]
        self.handle = None
        self.points = np.ascontiguousarray(cloud, dtype=np.float64)
        self.labels = np.ascontiguousarray(labels, dtype=np.int32)
        if (self.points.ndim != 2 or self.points.shape[1] != 3 or not len(self.points)
                or len(self.labels) != len(self.points) or self.labels.min() < 0
                or not np.isfinite(self.points).all()):
            raise ValueError('invalid labelled map for bounded kernel')
        lo = np.ascontiguousarray(self.points.min(0) - 1.0)
        hi = np.ascontiguousarray(self.points.max(0) + 1.0)
        if not np.isfinite(hi - lo).all() or np.any(hi - lo > 10000):
            raise ValueError('bounded kernel map extent exceeds safety bound')
        dimensions = np.maximum(4, np.ceil(hi - (lo - 1.0)).astype(np.int64) + 3)
        cells = int(np.prod(dimensions.astype(object)))
        if cells > 24000000 or len(self.points) > 4000000:
            raise ValueError('bounded kernel map exceeds safe index size')
        self.handle = self.lib.grid_build(self.ptr(self.points),
                                          self.labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                          len(self.points), self.ptr(lo), self.ptr(hi), 1.0)
        if not self.handle:
            raise MemoryError('bounded kernel index allocation failed')

    @staticmethod
    def ptr(array):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def query_allowed(self, placed, allowed):
        points = np.ascontiguousarray(placed, dtype=np.float64)
        mask = np.ascontiguousarray(allowed, dtype=np.uint8)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or self.labels.max() >= len(mask)):
            raise ValueError('invalid bounded-kernel query/mask')
        out = np.empty(len(points), dtype=np.float64)
        self.lib.grid_query(self.handle, self.ptr(points), len(points),
                            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), self.ptr(out))
        return out

    def close(self):
        if self.handle:
            self.lib.grid_free(self.handle)
            self.handle = None

    def __del__(self):
        if getattr(self, 'handle', None):
            self.close()

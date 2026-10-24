import ctypes
import sys
import time
from pathlib import Path

import numpy as np

import arbitration as ARB
from light_selection import SparseSelector


SOURCE = r'''
extern "C" __global__ void truncated_nn(
    const double* points, const long long* keys, long long n,
    const double* query, long long nq, const long long* origin,
    const long long* dims, double* output) {
    long long qi = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nq) return;
    double x = query[3*qi], y = query[3*qi+1], z = query[3*qi+2];
    long long cx = (long long)floor(x) - origin[0];
    long long cy = (long long)floor(y) - origin[1];
    long long cz = (long long)floor(z) - origin[2];
    double best = 1.0;
    for (int dx=-1; dx<=1; ++dx) {
      long long a = cx+dx;
      if (a<0 || a>=dims[0]) continue;
      for (int dy=-1; dy<=1; ++dy) {
        long long b = cy+dy;
        if (b<0 || b>=dims[1]) continue;
        for (int dz=-1; dz<=1; ++dz) {
          long long c = cz+dz;
          if (c<0 || c>=dims[2]) continue;
          long long key = (a*dims[1]+b)*dims[2]+c;
          long long lo=0, hi=n;
          while(lo<hi) {
            long long mid=lo+(hi-lo)/2;
            if(keys[mid]<key) lo=mid+1; else hi=mid;
          }
          for(long long j=lo; j<n && keys[j]==key; ++j) {
            double ex=x-points[3*j], ey=y-points[3*j+1], ez=z-points[3*j+2];
            double d=ex*ex+ey*ey+ez*ez;
            if(d<best) best=d;
          }
        }
      }
    }
    output[qi]=sqrt(best);
}
'''


class CudaKernel:
    def __init__(self):
        self.driver = ctypes.CDLL('libcuda.so.1')
        rtc = ctypes.CDLL(str(Path(sys.prefix) / 'lib/libnvrtc.so'))
        program = ctypes.c_void_p()
        rtc.nvrtcCreateProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                         ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        rtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        for name in ('nvrtcGetPTXSize', 'nvrtcGetProgramLogSize'):
            getattr(rtc, name).argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        for name in ('nvrtcGetPTX', 'nvrtcGetProgramLog'):
            getattr(rtc, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        rtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.check(rtc.nvrtcCreateProgram(ctypes.byref(program), SOURCE.encode(), b'selection.cu', 0, None, None))
        try:
            options = (ctypes.c_char_p * 2)(b'--gpu-architecture=compute_75', b'--fmad=false')
            status = rtc.nvrtcCompileProgram(program, 2, options)
            if status:
                size = ctypes.c_size_t()
                rtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                rtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError('NVRTC compile failed: ' + log.value.decode())
            size = ctypes.c_size_t()
            self.check(rtc.nvrtcGetPTXSize(program, ctypes.byref(size)))
            ptx = ctypes.create_string_buffer(size.value)
            self.check(rtc.nvrtcGetPTX(program, ptx))
        finally:
            rtc.nvrtcDestroyProgram(ctypes.byref(program))
        self.driver.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.driver.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.module, self.function = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.driver.cuModuleLoadData(ctypes.byref(self.module), ptx))
        self.check(self.driver.cuModuleGetFunction(ctypes.byref(self.function), self.module, b'truncated_nn'))

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError('CUDA/NVRTC status %d' % status)

    def query(self, points, keys, query, origin, dims, output, stream):
        values = [ctypes.c_void_p(points.data_ptr()), ctypes.c_void_p(keys.data_ptr()),
                  ctypes.c_longlong(len(points)), ctypes.c_void_p(query.data_ptr()),
                  ctypes.c_longlong(len(query)), ctypes.c_void_p(origin.data_ptr()),
                  ctypes.c_void_p(dims.data_ptr()), ctypes.c_void_p(output.data_ptr())]
        params = (ctypes.c_void_p * len(values))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in values])
        self.check(self.driver.cuLaunchKernel(self.function, (len(query)+127)//128, 1, 1,
                                              128, 1, 1, 0, ctypes.c_void_p(stream), params, None))


class GpuEvidence:
    def __init__(self, points, query, hypotheses, kernel, torch):
        started = time.monotonic()
        points = np.ascontiguousarray(points, dtype=np.float64)
        query = np.ascontiguousarray(query, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 405000
                or not 1 <= len(query) <= 2000 or not np.isfinite(points).all()
                or not np.isfinite(query).all() or np.max(np.abs(points)) > 1e6):
            raise ValueError('GPU selection evidence exceeds safety bounds')
        cells = np.floor(points).astype(np.int64)
        origin = cells.min(axis=0)
        dims = cells.max(axis=0) - origin + 1
        if np.prod(dims.astype(object)) >= 2**62:
            raise ValueError('GPU selection key extent exceeds safety bound')
        cells -= origin
        keys = (cells[:, 0]*dims[1]+cells[:, 1])*dims[2]+cells[:, 2]
        poses = [h.pose for h in hypotheses] + ARB.perturbations(hypotheses[0].pose)
        if len(poses) > 21:
            raise ValueError('GPU selection pose count exceeds safety bound')
        transformed = np.concatenate([query @ p[:3, :3].T + p[:3, 3] for p in poses])
        if not np.isfinite(transformed).all() or np.max(np.abs(transformed)) > 2e6:
            raise ValueError('invalid GPU selection transformed query')
        device = torch.device('cuda')
        prepared = time.monotonic()
        with torch.no_grad():
            gpu_points = torch.as_tensor(points, device=device)
            gpu_keys, order = torch.sort(torch.as_tensor(keys, device=device))
            counts = torch.unique_consecutive(gpu_keys, return_counts=True)[1]
            if counts.max().item() > 4096:
                raise ValueError('GPU selection cell occupancy exceeds safety bound')
            sorted_points = gpu_points.index_select(0, order)
            tensors = [sorted_points, gpu_keys] + [torch.as_tensor(np.ascontiguousarray(a), device=device)
                                                  for a in (transformed, origin, dims)]
            output = torch.empty(len(transformed), dtype=torch.float64, device=device)
            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before.record()
            kernel.query(*tensors, output, torch.cuda.current_stream().cuda_stream)
            after.record()
            distances = output.cpu().numpy().reshape(len(poses), len(query))
        self.scores = {np.asarray(p, dtype=np.float64).tobytes(): float(d.mean())
                       for p, d in zip(poses, distances)}
        self.n_map, self.base = len(points), query
        self.diagnostics = dict(gpu_prepare_seconds=prepared-started,
                                gpu_evidence_seconds=time.monotonic()-started,
                                gpu_kernel_seconds=before.elapsed_time(after)/1000)

    def score(self, pose, base=None):
        if base is not None:
            raise ValueError('GPU evidence requires the fixed shared query')
        return self.scores[np.asarray(pose, dtype=np.float64).tobytes()]


class GpuSelector(SparseSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch
        torch.cuda.init()
        torch.empty(1, device='cuda')
        self.torch, self.kernel = torch, CudaKernel()
        GpuEvidence(np.array([[0., 0., 0.], [1., 1., 1.]]), np.zeros((1, 3)),
                    [ARB.Hypothesis(np.eye(4), 'warmup')], self.kernel, torch)

    def evidence(self, points, query, hypotheses):
        return GpuEvidence(points, query, hypotheses, self.kernel, self.torch)

    def select(self, geometry, proposal, scan):
        pose, diagnostics, visual = super().select(geometry, proposal, scan)
        diagnostics['selection_mode'] = 'gpu_sparse'
        return pose, diagnostics, visual


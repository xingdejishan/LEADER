import ctypes
from pathlib import Path
import torch
import mixed_point_gradient as reference
import lean_mixed_hash as base
original_point_class=reference.PointGradient

library=ctypes.CDLL(str(Path(__file__).with_name('mixed_gradient_terms.so')))
library.mixed_gradient_terms.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*4+[ctypes.c_void_p]
library.mixed_gradient_terms.restype=ctypes.c_int


class FusedPointGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,points,tables,gradient,spacings):
        ctx.save_for_backward(points,tables,gradient,spacings)
        assert points.dtype==tables.dtype==gradient.dtype==spacings.dtype==torch.float32
        points=points.contiguous();tables=tables.contiguous();gradient=gradient.contiguous();spacings=spacings.contiguous()
        terms=points.new_empty((len(points),len(tables),8,tables.shape[-1],3))
        code=library.mixed_gradient_terms(points.data_ptr(),tables.data_ptr(),gradient.data_ptr(),spacings.data_ptr(),terms.data_ptr(),len(points),len(tables),tables.shape[1],tables.shape[-1],torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError('Coordinate mixed-gradient terms CUDA error '+str(code))
        return terms.sum((1,2,3))

    @staticmethod
    def backward(ctx,upstream):
        return original_point_class.backward(ctx,upstream)


def enable(model,module):
    base.enable(model,module)
    reference.PointGradient=FusedPointGradient
    return reference.enable(model,module)

import ctypes,types
from pathlib import Path
import torch
from torch.nn import functional as F
import fused_mixed_point as fused
import mixed_point_gradient as reference
import lean_mixed_hash as base

library=ctypes.CDLL(str(Path(__file__).with_name('mixed_backward_complete_v1.so')))
library.mixed_backward_complete.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*6+[ctypes.c_void_p]
library.mixed_backward_complete.restype=ctypes.c_int


class FusedPointGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,points,tables,gradient,spacings):
        return fused.FusedPointGradient.forward(ctx,points,tables,gradient,spacings)

    @staticmethod
    def backward(ctx,upstream):
        if torch.is_grad_enabled():return fused.original_point_class.backward(ctx,upstream)
        points,tables,gradient,spacings=ctx.saved_tensors
        points=points.contiguous();tables=tables.contiguous();gradient=gradient.contiguous();spacings=spacings.contiguous();upstream=upstream.contiguous()
        assert tables.shape[2]==2
        need_point,need_table,need_feature=ctx.needs_input_grad[:3]
        partial=points.new_empty((len(points),len(tables),3))
        table_gradient=torch.zeros_like(tables) if need_table else points.new_empty(1)
        feature_gradient=torch.empty_like(gradient)
        code=library.mixed_backward_complete(points.data_ptr(),tables.data_ptr(),gradient.data_ptr(),spacings.data_ptr(),upstream.data_ptr(),partial.data_ptr(),table_gradient.data_ptr(),feature_gradient.data_ptr(),len(points),len(tables),tables.shape[1],need_point,need_table,need_feature,torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError('Complete mixed backward CUDA error '+str(code))
        return partial.sum(1) if need_point else None,table_gradient if need_table else None,feature_gradient if need_feature else None,None


def enable(model,module):
    base.enable(model,module)
    reference.PointGradient=FusedPointGradient
    return reference.enable(model,module)

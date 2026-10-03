import ctypes
from pathlib import Path
import types
import torch
from torch.nn import functional as F

_library=None

def library():
    global _library
    if _library is None:
        _library=ctypes.CDLL(str(Path(__file__).resolve().parent/'hash_fused.so'))
        _library.hash_forward.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3+[ctypes.c_void_p]
        _library.hash_backward.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*3+[ctypes.c_void_p]
        _library.hash_forward.restype=_library.hash_backward.restype=ctypes.c_int
    return _library

class HashEncode(torch.autograd.Function):
    @staticmethod
    def forward(ctx,points,tables,spacings):
        assert points.is_cuda and points.dtype==tables.dtype==spacings.dtype==torch.float32
        assert tables.ndim==3 and tables.shape[2]==2 and spacings.numel()==tables.shape[0]
        points,tables,spacings=points.contiguous(),tables.contiguous(),spacings.contiguous()
        levels,size=tables.shape[:2]
        out=points.new_empty((len(points),levels*2))
        code=library().hash_forward(points.data_ptr(),tables.data_ptr(),spacings.data_ptr(),out.data_ptr(),len(points),levels,size,torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError('hash forward CUDA error %d'%code)
        ctx.save_for_backward(points,tables,spacings)
        return out

    @staticmethod
    def backward(ctx,grad):
        points,tables,spacings=ctx.saved_tensors
        grad=grad.contiguous()
        levels,size=tables.shape[:2]
        grad_table=torch.zeros_like(tables)
        partial=points.new_empty((len(points),levels,3))
        code=library().hash_backward(points.data_ptr(),tables.data_ptr(),spacings.data_ptr(),grad.data_ptr(),grad_table.data_ptr(),partial.data_ptr(),len(points),levels,size,torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError('hash backward CUDA error %d'%code)
        return partial.sum(1),grad_table,None

def forward(self,points):
    features=HashEncode.apply(points,self.tables,self.spacings)
    return F.normalize(self.mlp(features),dim=-1,eps=1e-8)

def enable(model):
    library()
    model.field.forward=types.MethodType(forward,model.field)
    return model

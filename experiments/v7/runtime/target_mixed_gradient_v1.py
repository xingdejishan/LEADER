import ctypes
from pathlib import Path
import torch
library=ctypes.CDLL(str(Path(__file__).with_name('target_mixed_backward_v1.so')))
library.target_mixed_backward.argtypes=[ctypes.c_void_p]*6
library.target_mixed_backward.restype=ctypes.c_int


def rotation_outputs(relative,scales):
    sine=torch.stack((relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]))*.5
    square=sine.square().sum()
    norm=square.clamp_min(1e-20).sqrt()
    cosine=(relative.trace()-1)*.5
    angle=torch.atan2(norm,cosine)
    small=(square<1e-8)&(cosine>0)
    value=.5*torch.where(small,square,angle.square())/scales[0].square()
    derivative=.5*(relative.trace()*torch.eye(3,device=relative.device,dtype=relative.dtype)-relative)
    norm_gradient=(derivative.T@sine)/norm*(square>=1e-20)
    gradient=angle*(cosine*norm_gradient+norm*sine)/(cosine.square()+norm.square())
    gradient=torch.where(small,derivative.T@sine,gradient)/scales[0]
    return value,gradient


class Target(torch.autograd.Function):
    @staticmethod
    def forward(ctx,relative,scales):
        ctx.save_for_backward(relative,scales)
        ctx.set_materialize_grads(False)
        return rotation_outputs(relative,scales)

    @staticmethod
    def backward(ctx,value_up,gradient_up):
        relative,scales=ctx.saved_tensors
        if torch.is_grad_enabled() or not relative.is_cuda or scales.requires_grad:
            with torch.enable_grad():
                outputs=rotation_outputs(relative,scales)
                value_up=torch.zeros_like(outputs[0]) if value_up is None else value_up
                gradient_up=torch.zeros_like(outputs[1]) if gradient_up is None else gradient_up
                inputs=(relative,scales) if scales.requires_grad else (relative,)
                result=torch.autograd.grad(outputs,inputs,(value_up,gradient_up),create_graph=torch.is_grad_enabled(),retain_graph=True)
                return result[0],result[1] if scales.requires_grad else None
        relative=relative.contiguous();scales=scales.contiguous()
        value_up=relative.new_zeros(()) if value_up is None else value_up.contiguous()
        gradient_up=relative.new_zeros(3) if gradient_up is None else gradient_up.contiguous()
        result=torch.empty_like(relative)
        assert relative.dtype==torch.float64 and scales.dtype==torch.float64 and relative.shape==(3,3)
        code=library.target_mixed_backward(relative.data_ptr(),scales.data_ptr(),value_up.data_ptr(),gradient_up.data_ptr(),result.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if code:raise RuntimeError('Target mixed backward CUDA error '+str(code))
        return result,None


def target_value_gradient(pose,truth,scales):
    relative=pose[:3,:3]@truth[:3,:3].T
    rotation,rotation_gradient=Target.apply(relative,scales)
    translation=pose[:3,3]-truth[:3,3]
    value=.5*translation.square().sum()/scales[3].square()+rotation
    return value,torch.cat((rotation_gradient,translation/scales[3]))

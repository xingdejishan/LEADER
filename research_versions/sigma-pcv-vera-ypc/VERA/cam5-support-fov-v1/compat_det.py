import torch

class InfrastructureFailure(Exception):
    pass

def install():
    original=torch.det
    def determinant(matrix):
        if matrix.is_cuda and matrix.shape[-2:]==(3,3):
            return original(matrix.cpu()).to(matrix.device)
        return original(matrix)
    torch.det=determinant
    return original

def fail_fast_pose(runtime):
    original=runtime.pose_from_correspondences
    def pose(*args,**kwargs):
        try:return original(*args,**kwargs)
        except RuntimeError as error:
            if 'nvrtc' in str(error).lower() or 'gpu-architecture' in str(error).lower():
                raise InfrastructureFailure(str(error)) from error
            raise
    runtime.pose_from_correspondences=pose

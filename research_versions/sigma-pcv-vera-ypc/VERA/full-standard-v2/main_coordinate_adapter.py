import torch
import MinkowskiEngine as ME

def aligned_sparse_cat(tensors,extra_sparse_tensors=[]):
    base=tensors[0]
    features=[base.F]
    for tensor in tensors[1:]:
        features.append(tensor.F if torch.equal(tensor.C,base.C) else tensor.features_at_coordinates(base.C.float()))
    features.extend(tensor.features_at_coordinates(base.C.float()) for tensor in extra_sparse_tensors)
    return ME.SparseTensor(coordinates=base.C,features=torch.cat(features,dim=1),coordinate_manager=base.coordinate_manager,tensor_stride=base.tensor_stride)

def activate():
    import models.model_mink as module
    module.MinkowskiSparseTensorCat=aligned_sparse_cat

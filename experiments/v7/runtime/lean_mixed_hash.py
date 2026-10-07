import ast
import contextlib
import types
import torch
from torch.nn import functional as F
from fused_field import HashEncode

coordinate_scope=0
original_class=None
point_backward=None


@contextlib.contextmanager
def coordinate_gradient_only():
    global coordinate_scope
    coordinate_scope+=1
    try:yield
    finally:coordinate_scope-=1


class LeanMixedHash(torch.autograd.Function):
    @staticmethod
    def forward(ctx,points,tables,spacings):
        return original_class.forward(ctx,points,tables,spacings)

    @staticmethod
    def backward(ctx,gradient):
        if torch.is_grad_enabled() and coordinate_scope:
            return point_backward(ctx,gradient)
        return original_class.backward(ctx,gradient)


def enable(model,module):
    global original_class,point_backward
    original_class=module.MixedHash
    tree=ast.parse(__import__('inspect').getsource(original_class))
    method=next(x for x in tree.body[0].body if isinstance(x,ast.FunctionDef) and x.name=='backward')
    method.decorator_list=[]
    body=[]
    for node in method.body:
        if isinstance(node,ast.If) and isinstance(node.body[0],ast.Return):continue
        if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ['contributions','table_gradient'] for x in node.targets):continue
        if isinstance(node,ast.Return):
            node.value=ast.Tuple(elts=[node.value.elts[0],ast.Constant(None),ast.Constant(None)],ctx=ast.Load())
        body.append(node)
    method.body=body
    generated=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(generated)
    namespace=dict(torch=torch,HashEncode=HashEncode)
    exec(compile(generated,'LeanMixedHash-point-only','exec'),namespace)
    point_backward=namespace['backward']

    def forward(self,points):
        return F.normalize(self.mlp(LeanMixedHash.apply(points,self.tables,self.spacings)),dim=-1,eps=1e-8)

    model.field.forward=types.MethodType(forward,model.field)
    return model

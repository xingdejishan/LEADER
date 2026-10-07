import ast
from pathlib import Path
import torch
import lean_mixed_hash

source_path=Path(__file__).with_name('adaptive_solver.py')
functions={'energy','value_gradient','volume','refine','loss','query'}


class Convert(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name=='query':
            return ast.parse('def query(model, points, chunk=16384):\n    return (yield points.float())\n').body[0]
        return self.generic_visit(node) if node.name in functions else node

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id in functions:
            return ast.YieldFrom(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='model' and node.func.attr=='field':
            assert len(node.args)==1 and not node.keywords
            return ast.Yield(node.args[0])
        if isinstance(node.func,ast.Attribute) and node.func.attr=='grad' and isinstance(node.func.value,ast.Attribute) and node.func.value.attr=='autograd':
            assert len(node.args)==2
            options=ast.Dict(keys=[ast.Constant(x.arg) for x in node.keywords],values=[x.value for x in node.keywords])
            return ast.Yield(ast.Tuple(elts=[ast.Constant('gradient'),*node.args,options],ctx=ast.Load()))
        return node


import template_moment_inverse
namespace={'__name__':'template_moment_reuse_generated','template_moment_inverse':template_moment_inverse}
source=source_path.read_text()
assert 'warp,inverse,_=stretching(point,pose,weight' in source
source=source.replace('warp,inverse,_=stretching(point,pose,weight','warp,inverse,gamma=stretching(point,pose,weight')
source=source.replace("    hs=[];ces=[];counts=[];gradients=[];size=settings['point_chunk']", "    moment_inverse=template_moment_inverse.inverse(point,pose,gamma,spacing,model.template.double())\n    hs=[];ces=[];counts=[];gradients=[];size=settings['point_chunk']")
source=source.replace("        denominator=torch.einsum('nki,nkj->nij',offsets,offsets)\n",'')
source=source.replace('torch.linalg.inv(denominator)','moment_inverse[start:start+len(central)]')
source=source.replace("        for iteration in range(settings['max_rounds']):\n            e,g,f,z,local=value_gradient(model,pose,context,scales,training)", "        current_derivative=None\n        for iteration in range(settings['max_rounds']):\n            if current_derivative is None:current_derivative=value_gradient(model,pose,context,scales,training)\n            e,g,f,z,local=current_derivative")
source=source.replace("            trial_e,trial_g,trial_f,_,_=value_gradient(model,trial,context,scales,training)", "            trial_derivative=value_gradient(model,trial,context,scales,training)\n            trial_e,trial_g,trial_f,_,_=trial_derivative")
source=source.replace('            if accepted:pose=trial;rejects=0', '            if accepted:pose=trial;rejects=0;current_derivative=trial_derivative')
source=source.replace("            trial_derivative=value_gradient(model,trial,context,scales,training)\n            trial_e,trial_g,trial_f,_,_=trial_derivative", "            trial_e,trial_f=energy(model,trial,context)\n            trial_e=trial_e.detach();trial_f=trial_f.detach()")
source=source.replace("            accepted=ratio is not None and ratio>settings['accept_ratio'] and float(actual.detach())>0", "            accepted=ratio is not None and ratio>settings['accept_ratio'] and float(actual.detach())>0\n            trial_derivative=value_gradient(model,trial,context,scales,training) if accepted else None\n            trial_g=trial_derivative[1] if accepted else g")
import analytic_pose_gradient_v1 as analytic_pose
namespace['target_gradient']=analytic_pose.target_gradient
original_tree=ast.parse(source)
old_gradient=next(x for x in original_tree.body if isinstance(x,ast.FunctionDef) and x.name=='value_gradient')
lines=source.splitlines(keepends=True)
source=''.join(lines[:old_gradient.lineno-1])+analytic_pose.VALUE_GRADIENT+''.join(lines[old_gradient.end_lineno:])
source=source.replace('torch.autograd.grad(target,z,create_graph=True,retain_graph=True)[0]','target_gradient(local,truth.double(),scales)')
assert 'torch.autograd.grad(target,z' not in source
source=source.replace('            trial_e,trial_f=energy(model,trial,context)', '            with torch.no_grad():trial_e,trial_f=energy(model,trial,context)')
import target_mixed_gradient_v1
namespace['target_value_gradient']=target_mixed_gradient_v1.target_value_gradient
source=source.replace('target=task_error(local,truth.double(),scales)\n            target_g=target_gradient(local,truth.double(),scales)','target,target_g=target_value_gradient(local,truth.double(),scales)')
source=source.replace('target=task_error(local,truth.double(),scales)\n                target_g=target_gradient(local,truth.double(),scales)','target,target_g=target_value_gradient(local,truth.double(),scales)')
assert 'target_g=target_gradient' not in source
tree=Convert().visit(ast.parse(source))
ast.fix_missing_locations(tree)
exec(compile(tree,str(source_path)+'[batched-fields-and-gradients]','exec'),namespace)
import compact_so3_backward_v1
namespace['so3_exp']=compact_so3_backward_v1.so3_exp


def loss_batch(model,frames,observations,settings,shape_enabled=True,query_chunk=262144):
    generators=[namespace['loss'](model,frame,observation,settings,shape_enabled) for frame,observation in zip(frames,observations)]
    assert len(generators)==len(frames)==len(observations)
    waiting={};results=[None]*len(frames)

    def advance(index,reply=None,initial=False):
        try:waiting[index]=next(generators[index]) if initial else generators[index].send(reply)
        except StopIteration as done:results[index]=done.value

    for index in range(len(frames)):advance(index,initial=True)
    while waiting:
        requests=waiting;waiting={}
        fields=[i for i,x in requests.items() if torch.is_tensor(x)]
        derivatives=[i for i,x in requests.items() if not torch.is_tensor(x)]
        if fields:
            for index in fields:
                points=requests[index]
                reply=model.field(points.float())
                advance(index,reply)
        if derivatives:
            options=requests[derivatives[0]][3]
            assert all(requests[i][0]=='gradient' and requests[i][3]==options for i in derivatives)
            outputs=sum(requests[i][1] for i in derivatives)
            inputs=tuple(requests[i][2] for i in derivatives)
            with lean_mixed_hash.coordinate_gradient_only():
                gradients=torch.autograd.grad(outputs,inputs,**options)
            for index,gradient in zip(derivatives,gradients):advance(index,(gradient,))
    assert all(x is not None for x in results)
    return results

import unittest

import torch

from models.far import AppearanceField,ForwardAppearanceRegistration,so3_exp


class FarTests(unittest.TestCase):
    def test_field_position_gradient(self):
        field=AppearanceField(4,8).double()
        x=torch.tensor([[.231,.521,-.41],[.812,-.321,1.212]],dtype=torch.float64,requires_grad=True)
        result=field(x)
        torch.testing.assert_close(result.norm(dim=1),torch.ones(2,dtype=torch.float64))
        result[:,0].sum().backward()
        self.assertGreater(float(x.grad.norm()),0)
        self.assertGreater(float(field.tables.grad.norm()),0)

    def test_shared_rigid_solve_sign_and_uniform(self):
        model=ForwardAppearanceRegistration(3,8).double()
        points=torch.tensor([[4.,0,0],[0,4.,0],[0,0,4.],[-4.,0,0],[0,-4.,0],[0,0,-4.]],dtype=torch.float64)
        pivot=points.new_zeros(3)
        expected=points.new_tensor([.01,-.02,.015,.1,-.1,.05])
        from models.far import skew
        j=torch.cat((-skew(points),torch.eye(3,dtype=points.dtype).expand(len(points),3,3)),-1)
        mu=torch.einsum('nik,k->ni',j,expected)
        cov=torch.eye(3,dtype=points.dtype).expand(len(points),3,3)*1e-6
        actual=model.solve(points,pivot,mu,cov)
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=1e-4)
        torch.testing.assert_close(model.solve(points,pivot,torch.zeros_like(mu),cov),torch.zeros(6,dtype=points.dtype))

    def test_two_rounds_gradients_and_so3(self):
        torch.manual_seed(20)
        model=ForwardAppearanceRegistration(3,8,8).double()
        model.raw_temperature.data.fill_(-3)
        points=torch.randn(8,3,dtype=torch.float64)
        descriptor=torch.nn.functional.normalize(torch.randn(8,32,dtype=torch.float64),dim=-1).requires_grad_()
        initial=torch.eye(4,dtype=torch.float64)
        truth=initial.clone()
        truth[:3,3]=truth.new_tensor([.05,.03,.02])
        pose,losses,stages=model(points,descriptor,initial,truth)
        (sum(losses)+(pose-truth).square().sum()).backward()
        self.assertGreater(float(model.field.tables.grad.norm()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        r=so3_exp(torch.zeros(3,dtype=torch.float64,requires_grad=True))
        torch.testing.assert_close(r,torch.eye(3,dtype=torch.float64))
        r.sum().backward()


if __name__=='__main__':
    unittest.main()

import unittest

import torch

from model import FactorialFusion


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.lidar=torch.randn(4,512)
        self.image=torch.randn(4,6,25,128)
        self.mask=torch.ones(4,6,25,dtype=torch.bool)
        self.direction=torch.randn(4,6,3)
        self.center=FactorialFusion('dedode_center')
        with torch.no_grad():
            self.center.fusion.residual[-1].weight.normal_()
        self.patch=FactorialFusion('dedode_patch')
        self.patch.load_state_dict(self.center.state_dict())

    def test_center_ignores_neighbors(self):
        before=self.center(self.lidar,self.image,self.mask,self.direction)
        self.image[:,:,:12]=float('nan')
        self.image[:,:,13:]=float('nan')
        torch.testing.assert_close(before,self.center(self.lidar,self.image,self.mask,self.direction),rtol=0,atol=0)

    def test_single_token_parity(self):
        self.mask.zero_()
        self.mask[:,:,12]=True
        torch.testing.assert_close(self.center(self.lidar,self.image,self.mask,self.direction),self.patch(self.lidar,self.image,self.mask,self.direction))

    def test_masked_nan_fallback(self):
        self.mask.zero_()
        self.image[:]=float('nan')
        for model in [self.center,self.patch]:
            torch.testing.assert_close(model(self.lidar,self.image,self.mask,self.direction),self.lidar,rtol=0,atol=0)

    def test_equal_initialization_and_parameters(self):
        states=[]
        for arm in ['dedode_center','dedode_patch','dino_center','dino_patch']:
            torch.manual_seed(2089)
            states.append(FactorialFusion(arm).state_dict())
        for key in states[0]:
            for state in states[1:]:
                self.assertTrue(torch.equal(states[0][key],state[key]))

    def test_all_parameters_active_in_both_modes(self):
        for model in [self.center,self.patch]:
            model(self.lidar,self.image,self.mask,self.direction).square().mean().backward()
            for name,p in model.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
                self.assertGreater(float(p.grad.abs().sum()),0,name)

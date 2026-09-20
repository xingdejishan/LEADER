import unittest

import torch

from model import MultiViewFusion


class Tests(unittest.TestCase):
    def data(self):
        return torch.randn(8,512),torch.randn(8,6,128),torch.ones(8,6,dtype=torch.bool),torch.randn(8,6,3)

    def test_cam5_ignores_other_views(self):
        l,x,m,d=self.data()
        model=MultiViewFusion('cam5')
        with torch.no_grad():
            model.fusion.residual[-1].weight.normal_()
        before=model(l,x,m,d)
        x[:,:5]=float('nan')
        self.assertTrue(torch.equal(before,model(l,x,m,d)))

    def test_mean_and_query_identical_with_one_visible_view(self):
        l,x,m,d=self.data()
        a,b=MultiViewFusion('mean'),MultiViewFusion('query')
        with torch.no_grad():
            a.fusion.residual[-1].weight.normal_()
        b.load_state_dict(a.state_dict())
        m[:,:5]=False
        torch.testing.assert_close(a(l,x,m,d),b(l,x,m,d))

    def test_missing_views_exact_identity_after_gate_changes(self):
        l,x,m,d=self.data()
        for arm in ['cam5','mean','query']:
            model=MultiViewFusion(arm)
            with torch.no_grad():
                model.fusion.residual[-1].weight.normal_()
            self.assertTrue(torch.equal(model(l,x*float('nan'),m&False,d),l))

    def test_mean_is_uniform_over_valid_views(self):
        l,x,m,d=self.data()
        m[:,3:]=False
        _,weights=MultiViewFusion('mean')(l,x,m,d,return_weights=True)
        torch.testing.assert_close(weights[:,:3],torch.full((8,3),1/3))
        self.assertEqual(float(weights[:,3:].sum()),0.)

    def test_same_initial_gate_across_arms(self):
        states=[]
        for arm in ['cam5','mean','query']:
            torch.manual_seed(2089)
            states.append(MultiViewFusion(arm).fusion.state_dict())
        for key in states[0]:
            self.assertTrue(all(torch.equal(s[key],states[0][key]) for s in states))


if __name__=='__main__':
    unittest.main()

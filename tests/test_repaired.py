import unittest
from unittest.mock import patch

import torch

from test_raybundle import scene
from raybundle.repaired import RepairedRayBundle,repaired_loss


class RepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_recovers_fifth_candidate_behind_appearance_top_four(self):
        frame,ray_map,_,_=scene()
        for name in ('origins','directions','descriptors'):
            frame[name]=frame[name][:1]
        frame['origins'].zero_()
        frame['directions']=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
        ray_map['mean']=torch.tensor([[0.,0.,-10.]]*4+[[0.,0.,10.]],dtype=torch.float64)
        ray_map['precision']=torch.eye(3,dtype=torch.float64).repeat(5,1,1)*1e4
        ray_map['descriptors']=frame['descriptors'].repeat(5,1)
        ray_map['descriptors'][4]+=.01
        model=RepairedRayBundle().double().eval()
        result=model(frame,ray_map,torch.tensor([[0,1,2,3]]))
        self.assertEqual(int(result['selected'][0]),4)
        self.assertTrue(all(4 in row for row in result['proposal_history']))

    def test_three_soft_updates_then_one_hard_update_and_refresh(self):
        frame,ray_map,_,_=scene()
        model=RepairedRayBundle().double().eval()
        with patch.object(model,'_hard_assignment',wraps=model._hard_assignment) as hard, patch.object(model,'soft_assignment',wraps=model.soft_assignment) as soft, patch.object(model,'retrieve',wraps=model.retrieve) as retrieval:
            model(frame,ray_map)
            self.assertEqual((soft.call_count,hard.call_count,retrieval.call_count),(3,1,4))

    def test_zero_visual_preserves_leader_even_with_asymmetric_targets(self):
        frame,ray_map,_,_=scene()
        frame['target'][:10,0]+=.4
        model=RepairedRayBundle().double().eval()
        def null(scores,indices,cameras):
            result=torch.zeros_like(scores)
            result[:,-1]=1
            return result
        with patch.object(model,'_hard_assignment',side_effect=null):
            result=model(frame,ray_map)
        self.assertTrue(torch.equal(result['pose'],frame['T0']))
        _,gradient=model.lidar_system(frame,frame['T0'])
        self.assertTrue(torch.equal(gradient,torch.zeros_like(gradient)))

    def test_soft_capacity_and_identity_loss_have_finite_gradients(self):
        frame,ray_map,_,truth=scene()
        model=RepairedRayBundle(rounds=2).double().train()
        scores=torch.tensor([[10.,9.,0.],[10.,9.,0.]],dtype=torch.float64,requires_grad=True)
        indices=torch.tensor([[0,1],[0,1]])
        p=model.soft_assignment(scores,indices,torch.zeros(2,dtype=torch.long))
        self.assertTrue(torch.all(p[:,:-1].sum(0)<=1+1e-12))
        self.assertTrue(torch.allclose(p.sum(1),torch.ones(2,dtype=torch.float64)))
        loss=repaired_loss(model,frame,ray_map,truth,torch.arange(8))
        loss.backward()
        self.assertGreater(float(model.adapter[-1].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))


if __name__=='__main__':
    unittest.main()

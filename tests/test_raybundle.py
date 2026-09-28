import unittest

import torch

from raybundle.geometry import (bearing_linearization, lidar_normal, normal_system,
                                predictive_score, retract)
from raybundle.model import RayBundleCorrespondence


def scene():
    torch.manual_seed(9)
    dtype = torch.float64
    n = 8
    source = torch.randn(40, 3, dtype=dtype) * 5
    mean = torch.randn(n, 3, dtype=dtype) * 2
    mean[:, 2] += 12
    truth = torch.eye(4, dtype=dtype)
    truth[:3, 3] = torch.tensor([.08, -.02, .01], dtype=dtype)
    origins = torch.zeros(n, 3, dtype=dtype)
    origins[:, 0] = .3
    directions = torch.nn.functional.normalize(mean - truth[:3, 3] - origins, dim=-1)
    descriptors = torch.randn(n, 128, dtype=dtype)
    reference = torch.cat((descriptors, descriptors + .08 * torch.randn_like(descriptors)), 0)
    shift = torch.tensor([1., -.3, .5], dtype=dtype)
    ray_map = dict(mean=torch.cat((mean, mean + shift), 0), precision=torch.eye(3, dtype=dtype).repeat(n*2, 1, 1) * 1e4,
                   descriptors=reference)
    frame = dict(source=source, target=source.clone(), weights=torch.ones(len(source), dtype=dtype),
                 T0=torch.eye(4,dtype=dtype), origins=origins, directions=directions, descriptors=descriptors,
                 independent_sigma=torch.tensor(.1), shared_covariance=torch.diag(torch.tensor([.2**2]*3+[.04**2]*3,dtype=dtype)))
    candidates = torch.stack((torch.arange(n)+n, torch.arange(n)), -1)
    return frame, ray_map, candidates, truth


class RayBundleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_bearing_jacobians_and_camera_origin(self):
        frame, ray_map, _, _ = scene()
        pose = retract(frame['T0'], torch.tensor([.1,.2,-.1,.03,-.02,.01],dtype=torch.float64))
        point = ray_map['mean'][:1]
        origin, direction = frame['origins'][:1], frame['directions'][:1]
        r,j,d,_ = bearing_linearization(pose, point, origin, direction)
        for k in range(6):
            delta = torch.zeros(6,dtype=torch.float64)
            delta[k] = 1e-6
            plus = bearing_linearization(retract(pose,delta),point,origin,direction)[0]
            minus = bearing_linearization(retract(pose,-delta),point,origin,direction)[0]
            self.assertTrue(torch.allclose((plus-minus)/2e-6,j[...,k],atol=1e-7))
        for k in range(3):
            delta = torch.zeros(3,dtype=torch.float64)
            delta[k] = 1e-6
            plus = bearing_linearization(pose,point+delta,origin,direction)[0]
            minus = bearing_linearization(pose,point-delta,origin,direction)[0]
            self.assertTrue(torch.allclose((plus-minus)/2e-6,d[...,k],atol=1e-7))
        zero_origin = bearing_linearization(pose,point,origin*0,direction)[0]
        self.assertGreater(float((zero_origin-r).abs().max()), .001)

    def test_reassociate_and_output_real_identities(self):
        frame, ray_map, candidates, truth = scene()
        model = RayBundleCorrespondence(rounds=3).double().eval()
        result = model(frame,ray_map,candidates)
        self.assertTrue(torch.equal(result['selected'],torch.arange(8)))
        self.assertLess(float((result['pose'][:3,3]-truth[:3,3]).norm()),float(truth[:3,3].norm()))
        self.assertTrue(torch.equal(result['candidates'],candidates))
        self.assertEqual(len(torch.unique(result['selected'])),8)

    def test_finite_gradients_two_updates(self):
        from raybundle.cli import train_step
        frame, ray_map, candidates, truth = scene()
        model = RayBundleCorrespondence(rounds=2).double()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        for _ in range(2):
            _, gradient = train_step(model,optimizer,frame,ray_map,candidates,truth,torch.ones(8,dtype=torch.long))
            self.assertGreater(gradient,0)
        self.assertGreater(float(model.adapter[0].weight.grad.norm()),0)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_weak_depth_reduces_directional_information(self):
        frame, ray_map, candidates, _ = scene()
        ids = candidates[:,1:]
        r,j,d,_ = bearing_linearization(frame['T0'],ray_map['mean'][ids],frame['origins'][:,None],frame['directions'][:,None])
        lh,lg = lidar_normal(frame['T0'],frame['source'],frame['target'],frame['weights'],.1,frame['shared_covariance'])
        weights = torch.ones_like(ids,dtype=torch.float64)
        strong = normal_system(lh,lg,ray_map['mean'],ray_map['mean'],ray_map['precision'],ids,r,j,d,weights,.005**2)
        weak = normal_system(lh,lg,ray_map['mean'],ray_map['mean'],ray_map['precision']*.000001,ids,r,j,d,weights,.005**2)
        self.assertGreater(float(torch.trace(weak[2])),float(torch.trace(strong[2])))

    def test_leave_group_out_removes_all_own_messages(self):
        frame, ray_map, candidates, _ = scene()
        r,j,d,_ = bearing_linearization(frame['T0'],ray_map['mean'][candidates],frame['origins'][:,None],frame['directions'][:,None])
        lh,lg = lidar_normal(frame['T0'],frame['source'],frame['target'],frame['weights'],.1,frame['shared_covariance'])
        mask = torch.ones(8,1,dtype=torch.float64)
        mask[:2] = 0
        base = torch.rand(8,2,dtype=torch.float64)
        alternate = base.clone()
        alternate[:2] = 100
        systems = [normal_system(lh,lg,ray_map['mean'],ray_map['mean'],ray_map['precision'],candidates,r,j,d,a*mask,.005**2) for a in (base,alternate)]
        scores = [predictive_score(s,candidates[:2],r[:2],j[:2],d[:2],.005**2) for s in systems]
        self.assertTrue(torch.equal(scores[0],scores[1]))

    def test_shared_lidar_error_does_not_shrink_with_duplicate_points(self):
        frame,_,_,_ = scene()
        hs=[]
        for repeat in (1,20):
            h,_ = lidar_normal(frame['T0'],frame['source'].repeat(repeat,1),frame['target'].repeat(repeat,1),frame['weights'].repeat(repeat),.1,frame['shared_covariance'])
            hs.append(h)
        self.assertLess(float(torch.trace(hs[1])/torch.trace(hs[0])),1.1)

    def test_missing_candidates_and_gt_isolation(self):
        frame, ray_map, candidates,_ = scene()
        model = RayBundleCorrespondence().double().eval()
        result = model(frame,ray_map,torch.full_like(candidates,-1))
        self.assertTrue(torch.equal(result['pose'],frame['T0']))
        with self.assertRaises(ValueError):
            model(dict(frame,T_WB=frame['T0']),ray_map,candidates)

    def test_unlabelled_or_missing_candidates_training_is_finite(self):
        from raybundle.cli import train_step
        frame, ray_map, candidates, truth = scene()
        model = RayBundleCorrespondence().double()
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
        ignored = torch.full((len(candidates),),-100,dtype=torch.long)
        for ids in (candidates,torch.full_like(candidates,-1)):
            loss, gradient = train_step(model,optimizer,frame,ray_map,ids,truth,ignored)
            self.assertTrue(torch.isfinite(torch.tensor([loss,gradient])).all())

    def test_far_landmark_retrieval_uses_reference_camera_location(self):
        from raybundle.io import candidate_indices
        frame, ray_map, _, _ = scene()
        ray_map['mean'] = ray_map['mean'] + 1000
        ray_map['reference_centers'] = torch.zeros_like(ray_map['mean'])
        ray_map['reference_radii'] = torch.zeros(len(ray_map['mean']),dtype=torch.float64)
        ids = candidate_indices(RayBundleCorrespondence().double(),frame,ray_map,{}, {})
        self.assertTrue(torch.all(ids>=0))


if __name__ == '__main__':
    unittest.main()

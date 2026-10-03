import copy,unittest

import torch
from torch.nn import functional as F

from models.far import AppearanceField
from models.far_covolume import CoVolumeFAR,tukey,registration_loss_covolume,registration_loss_batch


class CoVolumeTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(20)
        model=CoVolumeFAR(point_chunk=32)
        model.field=AppearanceField(levels=2,table_bits=4)
        return model

    def test_tukey_flat_tail_and_zero_gradient(self):
        square=torch.tensor([.2,2.,4.],requires_grad=True)
        loss=tukey(square,1.2)
        self.assertEqual(float(loss[1]),float(loss[2]))
        loss.sum().backward()
        self.assertGreater(float(square.grad[0]),0)
        self.assertEqual(float(square.grad[1]),0)

    def test_visible_geometry_is_not_counted_again(self):
        model=self.model().double()
        points=torch.randn(12,3,dtype=torch.float64)
        pose=torch.eye(4,dtype=torch.float64)
        mean=torch.full_like(points,.04)
        covariance=torch.eye(3,dtype=torch.float64).expand(12,3,3)*.1
        structure=torch.eye(3,dtype=torch.float64).expand(12,3,3)/3
        visible=torch.arange(12)
        a,_=model.solve_shared(points,pose,mean,covariance,points,points+.5,structure,visible,0)
        b,_=model.solve_shared(points,pose,mean,covariance,points,points-.5,structure,visible,0)
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_structure_relaxes_tangent_and_rotates_with_pose(self):
        model=self.model().double()
        pose=torch.eye(4,dtype=torch.float64)
        structure=torch.diag(torch.tensor([.5,.5,0.],dtype=torch.float64))[None]
        precision,_=model.geometry_precision(pose,structure,0)
        self.assertLess(float(precision[0,0,0]),float(precision[0,2,2]))
        pose[:3,:3]=torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]],dtype=torch.float64)
        rotated,_=model.geometry_precision(pose,structure,0)
        self.assertGreater(float(rotated[0,0,0]),float(rotated[0,2,2]))

    def test_pose_loss_trains_both_stage_geometry_parameters(self):
        model=self.model()
        points=torch.randn(20,3)*2
        initial=torch.eye(4)
        initial[:3,3]=torch.tensor([.1,-.06,.04])
        truth=torch.eye(4)
        target=points+torch.tensor([.04,.02,-.03])
        structure=torch.diag(torch.tensor([.5,.4,.1]))[None].expand(20,3,3)
        visible=torch.cat((torch.arange(10),torch.full((10,),-1,dtype=torch.long)))
        observed=F.normalize(torch.randn(20,32),dim=-1)
        loss,_=registration_loss_covolume(model,points,points,observed,truth,initial,points,target,structure,visible)
        loss.backward()
        for name in ('raw_structure','raw_geo_sigma','raw_sigma','raw_temperature'):
            gradient=getattr(model,name).grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient.abs()>0).all(),name)

    def test_batched_observation_matches_individual_frames(self):
        model=self.model()
        other=copy.deepcopy(model)
        sams=[torch.randn(256,8,8) for _ in range(3)]
        rgbs=[torch.randn(3,64,80) for _ in range(3)]
        pixels=[torch.rand(count,2)*torch.tensor([80.,64.]) for count in (9,13,7)]
        sizes=[(819,1024)]*3
        individual=[model.observation(sam,rgb,pixel,size) for sam,rgb,pixel,size in zip(sams,rgbs,pixels,sizes)]
        batched=other.observation_batch(sams,rgbs,pixels,sizes)
        for a,b in zip(individual,batched):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        sum(value.square().sum()+value.sum() for value in individual).backward()
        sum(value.square().sum()+value.sum() for value in batched).backward()
        for (name,a),(_,b) in zip(model.named_parameters(),other.named_parameters()):
            if a.grad is None:self.assertIsNone(b.grad,name)
            else:torch.testing.assert_close(a.grad,b.grad,atol=1e-4,rtol=1e-4)

    def test_batch_loss_and_gradients_equal_mean_of_frames(self):
        model=self.model()
        other=copy.deepcopy(model)
        points=[torch.randn(count,3)*2 for count in (12,17,9)]
        observations=[F.normalize(torch.randn(len(point),32),dim=-1) for point in points]
        initials=[torch.eye(4) for _ in points]
        for pose in initials:pose[:3,3]=torch.tensor([.1,-.06,.04])
        truths=[torch.eye(4) for _ in points]
        targets=[point+torch.tensor([.04,.02,-.03]) for point in points]
        structures=[torch.diag(torch.tensor([.5,.4,.1]))[None].expand(len(point),3,3) for point in points]
        visible=[torch.cat((torch.arange(len(point)//2),torch.full((len(point)-len(point)//2,),-1,dtype=torch.long))) for point in points]
        individual=[registration_loss_covolume(model,point,point,observed,truth,initial,point,target,structure,ids)[0] for point,observed,truth,initial,target,structure,ids in zip(points,observations,truths,initials,targets,structures,visible)]
        mean=torch.stack(individual).mean()
        batched,_=registration_loss_batch(other,points,points,observations,truths,initials,points,targets,structures,visible)
        torch.testing.assert_close(mean,batched,atol=1e-6,rtol=1e-5)
        mean.backward()
        batched.backward()
        for (name,a),(_,b) in zip(model.named_parameters(),other.named_parameters()):
            if a.grad is None:self.assertIsNone(b.grad,name)
            else:torch.testing.assert_close(a.grad,b.grad,atol=1e-5,rtol=1e-3)


if __name__=='__main__':unittest.main()

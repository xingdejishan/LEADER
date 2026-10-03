import argparse,json
from pathlib import Path

import torch

from models.far import ForwardAppearanceRegistration,so3_exp
from tools.far_runtime import setup,make_dataset,load_frame,observation,verify_sources
from tools.magic_nclt_runtime import load_pretrained_model,safe_json_dump,state_dict_sha256
from tools.predict_far import lidar_prediction,backend


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    output=Path(args.output)
    output.mkdir(exist_ok=False)
    setup(output)
    verify_sources(config)
    device=torch.device('cuda:0')
    dataset=make_dataset(config,'dev')
    leader,checkpoint=load_pretrained_model(config['base_checkpoint'],False,device)
    leader.eval()
    initial_hash=state_dict_sha256(leader.state_dict())
    torch.manual_seed(config['seed'])
    far=ForwardAppearanceRegistration().to(device).eval()
    with torch.no_grad():
        sample,points,pixels,rgb=load_frame(dataset,0,device)
        assert 'T_WB' not in sample
        pose,all_points,prediction,indices=lidar_prediction(leader,sample,device,config['seed'])
        descriptor=observation(far,sample,pixels,rgb,device)
        refined,_,stages=far(points,descriptor,pose)
        initial_shift=float((refined[:3,3]-pose[:3,3]).norm())
        initial_rotation=float((refined[:3,:3]-pose[:3,:3]).norm())
        assert initial_shift<.002 and initial_rotation<.002,(initial_shift,initial_rotation)
        r=so3_exp(pose.new_tensor([.001,-.002,.003]))
        pivot=pose[:3,3]
        shift=pose.new_tensor([.08,-.04,.03])
        translation=pivot-r@pivot+shift
        corrected_targets=prediction[:,:3]@r.T+translation
        final=backend(all_points[indices],corrected_targets[indices],config['seed'])
        expected=pose.clone()
        expected[:3,:3]=r@pose[:3,:3]
        expected[:3,3]=r@pose[:3,3]+translation
        position_error=float((expected[:3,3]-final[:3,3]).norm())
        rotation_error=float((expected[:3,:3]-final[:3,:3]).norm())
        assert position_error<.002 and rotation_error<.002,(position_error,rotation_error)
        assert state_dict_sha256(leader.state_dict())==initial_hash
    safe_json_dump(output/'passed.json',dict(ground_truth_read_online=False,leader_unchanged=True,visible_points=len(points),
        initial_shift_m=initial_shift,initial_rotation_matrix_difference=initial_rotation,
        backend_equivariance_position_m=position_error,backend_equivariance_rotation_matrix_difference=rotation_error,
        gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))


if __name__=='__main__':main()

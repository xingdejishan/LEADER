import json
from pathlib import Path
import numpy as np
import run


def main():
    roots = {name: Path('/home/zhang/leader-image-gate-'+suffix) for name, suffix in [('line3', 'context'), ('line4', 'geometry')]}
    rows = json.loads((roots['line4']/'manifest.json').read_text())
    results = {}
    for split in ['train', 'val']:
        results[split] = {}
        for method, root in roots.items():
            records = []
            for row in [r for r in rows if r['split']==split]:
                with np.load(root/'visual'/(row['frame_id']+'.npz')) as data:
                    valid = data['valid']
                    coordinates = data['image'][valid, -6:].reshape(-1, 2, 3)*100
                with np.load(root/'lidar'/(row['frame_id']+'.npz')) as data:
                    body_pose = data['GT']
                with np.load(Path('/home/zhang/leader-image-gate/projection_audit/mapping')/(row['frame_id']+'.npz')) as data:
                    raw = data['projection_xyz'][valid]
                camera_pose = np.loadtxt(row['pose'])
                true_world = raw @ body_pose[:3, :3].T + body_pose[:3, 3]
                true_camera = (true_world-camera_pose[:3, 3]) @ camera_pose[:3, :3]
                predicted_camera = (coordinates-camera_pose[:3, 3]) @ camera_pose[:3, :3]
                records.append(dict(frame_id=row['frame_id'], count=int(valid.sum()),
                    median_relative_depth=np.median(abs(predicted_camera[:, :, 2]-true_camera[:, None, 2])/np.maximum(true_camera[:, None, 2], .1), axis=0).tolist(),
                    median_world_error=np.median(np.linalg.norm(coordinates-true_world[:, None], axis=-1), axis=0).tolist(),
                    negative_depth_fraction=(predicted_camera[:, :, 2]<=0).mean(0).tolist()))
            results[split][method] = dict(frames=len(records),
                mean_frame_median_relative_depth=np.mean([r['median_relative_depth'] for r in records], axis=0).tolist(),
                mean_frame_median_world_error=np.mean([r['median_world_error'] for r in records], axis=0).tolist(), records=records)
    results['definition'] = 'Order coarse, refined; actual representative raw-point GT used only for post-training diagnostics. Not used for retrieval/inference. Not final pose accuracy.'
    run.save_json(roots['line4']/'geometry_diagnostic.json', results)
    run.save_json(run.HERE/'results/geometry_retrain/geometry_diagnostic.json', results)
    print(json.dumps({k: {m: {n:v for n,v in d.items() if n!='records'} for m,d in results[k].items()} for k in ['train', 'val']}, indent=2))


if __name__ == '__main__':
    main()

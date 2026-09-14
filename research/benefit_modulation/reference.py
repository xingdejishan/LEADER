import json
import torch
from experiment import ARGS, OUT, run, selected, pose_error, Matcher
from fusion import ImageGate

torch.set_num_threads(4)
decoder = run.load_leader(ARGS).decoder
head = ImageGate().cuda()
head.load_state_dict(torch.load(ARGS.output / 'aligned.pt'))
matcher = Matcher(inlier_threshold=2., d_thre=2, num_iterations=10, ratio=.15, nms_radius=.1, max_points=3000, k1=30)
rows = json.loads((ARGS.output / 'manifest.json').read_text())
records = []
with torch.no_grad():
    for row in sorted([r for r in rows if r['split'] == 'val'], key=lambda r: r['frame_id']):
        item = run.frame(ARGS, row)
        item['indices'] = selected(item['prediction'])
        pred = decoder(head(item['features'], item['image'], item['valid']))
        delta = (pred[:, :3] - item['target']).norm(dim=-1) - (item['prediction'][:, :3] - item['target']).norm(dim=-1)
        mask = torch.zeros_like(item['valid'])
        mask[item['indices']] = True
        mask &= item['valid']
        records.append(dict(frame_id=row['frame_id'], standard=pose_error(item, pred, matcher),
                            fixed=pose_error(item, pred, matcher, True),
                            protected=dict(count=int(mask.sum()), delta_sum=float(delta[mask].sum()),
                                           harmed=int((delta[mask] > .01).sum()), helped=int((delta[mask] < -.01).sum()))))
run.save_json(OUT / 'reference_line1.json', records)
print(run.metrics([r['standard'] for r in records]))

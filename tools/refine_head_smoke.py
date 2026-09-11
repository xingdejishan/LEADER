# Smoke test for the refinement-head experiment (v1-residual-head).
# Verifies: checkpoint load compatibility, zero-init equivalence with the
# baseline, forward_with_feature consistency, gradient isolation, and that the
# frozen backbone's BN stats are untouched by a refinement-only train step.
import sys
import numpy as np
import torch
import MinkowskiEngine as ME

sys.path.insert(0, '/root/rivermind-data/LEADER-v1-residual-head')

from models.model_mink import LEADER, RefinementHead

CKPT = '/root/rivermind-data/LEADER/checkpoints/checkpoint_epoch_49'
DEVICE = 'cuda'
torch.manual_seed(0)
np.random.seed(0)

def make_batch(n_points=40000, bs=2, res=1024):
    coords_list, feats_list = [], []
    for b in range(bs):
        c = np.random.randint(0, (res, 128, 128), size=(n_points, 3)).astype(np.int32)
        c = np.concatenate([np.full((n_points, 1), b, dtype=np.int32), c], axis=1)
        coords_list.append(torch.from_numpy(c))
        feats_list.append(torch.rand(n_points, 3).float())
    coords = torch.cat(coords_list, 0)
    feats = torch.cat(feats_list, 0)
    return ME.SparseTensor(feats.to(DEVICE), coords.to(DEVICE))

def report(name, ok, detail=''):
    print(('PASS' if ok else 'FAIL'), name, detail)
    if not ok:
        sys.exit(1)

model = LEADER(in_channels=3, out_channels=4, feat_channels=512,
               width=1024, use_refinement=True).to(DEVICE)

# 1. load frozen backbone: only refinement_head keys may be missing
from safetensors.torch import load_file
state = load_file(CKPT + '/model.safetensors')
missing, unexpected = model.load_state_dict(state, strict=False)
report('load_backbone_missing_only_head',
       unexpected == [] and missing and all(k.startswith('refinement_head.') for k in missing),
       f'missing={len(missing)} unexpected={len(unexpected)}')

# 2. param count
n = sum(p.numel() for p in model.refinement_head.parameters())
report('head_param_count', n == 66051, str(n))

for p in model.parameters():
    p.requires_grad = False
for p in model.refinement_head.parameters():
    p.requires_grad = True

model.eval()
model.refinement_head.train()

input_sp = make_batch()

with torch.no_grad():
    enc = model.encoder(input_sp)
    enc_F = enc.F

    h, coarse_out = model.decoder.forward_with_feature(enc_F)
    pred_direct = model.decoder(enc_F)

    # 3. forward_with_feature consistent with forward
    report('forward_with_feature_consistent', torch.allclose(pred_direct, coarse_out, atol=1e-6))

    # 4. h is exactly pred_out[:-1] applied to the post-block features
    input_x = model.decoder.proj_in(enc_F)
    for block in model.decoder.blocks:
        mlp_x = block["mlp"](input_x)
        mlp_max = mlp_x.view(*mlp_x.shape[:-1], 512, 4).max(dim=-1)[0]
        input_x = model.decoder.relu(block["norm"](input_x + mlp_max))
    h_ref = model.decoder.pred_out[:3](input_x)
    report('h_is_pred_out_prefix', torch.allclose(h, h_ref, atol=1e-6))

    # 5. zero-init head: delta == 0 and refined output == baseline output
    delta = model.refinement_head(h)
    report('zero_init_delta_is_zero', float(delta.abs().max()) == 0.0)
    pred_f = torch.cat([coarse_out[:, :3] + delta, coarse_out[:, 3:4]], dim=1)
    report('refined_equals_baseline', torch.allclose(pred_f, pred_direct, atol=1e-6))

    bn_before = {k: v.clone() for k, v in model.state_dict().items() if 'running' in k or '.bn' in k}

# 6. training step: gradient isolation + optimizer updates only the head
opt = torch.optim.Adam(model.refinement_head.parameters(), lr=1e-3)
w_enc_before = model.encoder.stem[0].linear.weight.detach().clone()
w_dec_before = model.decoder.pred_out[0].weight.detach().clone()

delta = model.refinement_head(h)
pred_f = torch.cat([coarse_out[:, :3] + delta, coarse_out[:, 3:4]], dim=1)

batch_idx = enc.C[:, 0].long()
gt = pred_f[:, :3].detach() + 0.05  # pretend GT shifted 5cm
l_raw = (pred_f[:, :3] - gt).norm(dim=-1)
loss = l_raw.mean()
loss.backward()
opt.step()

enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
dec_grads = [p.grad for p in model.decoder.parameters() if p.grad is not None]
head_fc2_grad = model.refinement_head.fc2.weight.grad
head_fc1_grad = model.refinement_head.fc1.weight.grad
report('no_grad_leaks_to_encoder', len(enc_grads) == 0, f'{len(enc_grads)} encoder grads')
report('no_grad_leaks_to_decoder', len(dec_grads) == 0, f'{len(dec_grads)} decoder grads')
report('fc2_grad_nonzero', head_fc2_grad is not None and float(head_fc2_grad.abs().sum()) > 0)
# zero-init final layer blocks gradient to fc1 on the first step (expected)
report('fc1_grad_zero_first_step', float(head_fc1_grad.abs().sum()) == 0.0)

report('encoder_weights_untouched', torch.equal(model.encoder.stem[0].linear.weight.detach(), w_enc_before))
report('decoder_weights_untouched', torch.equal(model.decoder.pred_out[0].weight.detach(), w_dec_before))
report('fc2_updated', not torch.equal(model.refinement_head.fc2.weight.detach(), torch.zeros_like(model.refinement_head.fc2.weight)))

# 7. BN running stats unchanged (backbone stayed in eval during the step)
bn_after = {k: v for k, v in model.state_dict().items() if 'running' in k or '.bn' in k}
report('bn_stats_unchanged', all(torch.equal(bn_before[k], bn_after[k]) for k in bn_before), f'{len(bn_before)} buffers checked')

print('ALL CHECKS PASSED')

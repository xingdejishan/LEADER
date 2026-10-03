import json
import math
from pathlib import Path

import torch

from repair import EndpointRepair, baseline_rotation, protected_pool, repair_supervision


def run():
    torch.set_num_threads(1)
    torch.manual_seed(20)
    n, m = 120, 7
    points = torch.randn(n, 3)
    angle = .73
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.], [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    translation = torch.tensor([3., -2., 1.])
    target = points @ rotation.T + translation
    base = torch.cat((target, torch.linspace(2., -2., n)[:, None]), -1)
    candidates = points[:, None] + torch.randn(n, m, 3) * .2
    features = torch.randn(n, 512)
    tokens = torch.randn(n, m, 64)
    valid = torch.ones(n, m, dtype=torch.bool)
    model = EndpointRepair()
    estimate, ok = baseline_rotation(points, base)
    assert ok and (estimate - rotation).abs().max() < 1e-5
    result = model(features, points, base, tokens, candidates, valid)
    assert torch.equal(result.anchors, points) and torch.equal(result.output, base)
    assert torch.equal(result.selected, base[:, 3].topk(n // 2).indices)
    assert not result.admitted.any() and result.swaps == 0
    transported_target = candidates @ rotation.T + translation
    assert (result.candidate_prediction - transported_target).abs().max() < 1e-5
    empty = model(features, points, base, tokens, candidates, torch.zeros_like(valid))
    assert torch.equal(empty.output, base) and torch.equal(empty.anchors, points)
    assert not empty.attention.any()
    same_tokens = tokens[:, :1].expand(-1, m, -1).clone()
    result = model(features, points, base, same_tokens, candidates, valid)
    assert torch.isfinite(result.attention).all()
    assert torch.equal(result.output, base)
    noisy_base = base.clone()
    noisy_base[:, :3] += torch.randn(n, 3) * .2
    result = model(features, points, noisy_base, tokens, candidates, valid, rotation, True)
    auxiliary, evidence = repair_supervision(result, noisy_base, points, rotation, translation)
    assert torch.isfinite(auxiliary)
    auxiliary.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.residual[-1].weight.grad.abs().sum() > 0
    assert model.quality_delta[-1].weight.grad.abs().sum() > 0
    assert model.base_quality[-1].weight.grad.abs().sum() > 0
    assert not result.admitted.any()
    errors = torch.full((n,), .5)
    proposed_errors = errors.clone()
    admitted = torch.zeros(n, dtype=torch.bool)
    outsiders = torch.arange(n // 2, n // 2 + 20)
    admitted[outsiders] = True
    proposed_errors[outsiders] = .1
    selected, protected, swaps = protected_pool(base, points, base, admitted, errors, proposed_errors)
    assert swaps == int(n // 2 * .25) and len(selected.unique()) == n // 2
    assert (base[:, 3].topk(n // 4).indices[:, None] == selected[None]).any(-1).all()
    proposed_errors[outsiders] = .7
    selected, _, swaps = protected_pool(base, points, base, admitted, errors, proposed_errors)
    assert swaps == 0 and torch.equal(selected, base[:, 3].topk(n // 2).indices)
    collinear = torch.zeros_like(points)
    collinear[:, 0] = torch.linspace(-1, 1, n)
    _, ok = baseline_rotation(collinear, base)
    assert not ok
    result = model(features, points, base, tokens, candidates, valid, rotation, False)
    assert not result.admitted.any() and torch.equal(result.output, base)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.residual[-1].weight.normal_(std=.1)
        model.quality_delta[-1].bias.fill_(-3.)
    result = model(features, points, noisy_base, tokens, candidates, valid, rotation, True)
    assert result.admitted.any()
    assert not result.admitted[result.protected].any()
    assert torch.equal(result.output[result.protected], noisy_base[result.protected])
    assert torch.equal(result.anchors[result.protected], points[result.protected])
    index = torch.arange(n)
    is_raw_candidate = ((result.anchors[:, None] - candidates).norm(dim=-1) < 1e-6).any(-1)
    assert is_raw_candidate[result.admitted].all()
    assert torch.equal(result.anchors[~result.admitted], points[~result.admitted])
    candidate_residual = result.candidate_prediction - noisy_base[:, None, :3] - (candidates - points[:, None]) @ rotation.T
    assert candidate_residual.norm(dim=-1).max() <= .50001
    auxiliary, evidence = repair_supervision(result, noisy_base, points, rotation, translation)
    auxiliary.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    query_gradient = model.query.weight.grad.abs().sum() if model.query.weight.grad is not None else torch.tensor(0.)
    assert query_gradient > 0 and model.key.weight.grad.abs().sum() > 0
    return {'passed': True, 'synthetic_cases': ['cold exact baseline identity', 'analytic nonidentity transport', 'empty visibility', 'constant visual tokens', 'candidate supervision independent of admission', 'quality pool admits predicted-better outsiders', 'reject worse outsiders', 'degenerate rotation fallback', 'protected pairs unchanged', 'measured candidate membership', 'bounded residual', 'attention and quality finite gradients'], 'parameter_count': sum(p.numel() for p in model.parameters()), 'query_gradient_l1': float(query_gradient), 'scope': 'Structure and autograd checks; no optimizer, training or accuracy claim'}


if __name__ == '__main__':
    print(json.dumps(run()))

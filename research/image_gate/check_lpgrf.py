import json
from pathlib import Path
import torch
from lpgrf import LoFTRLocal, LPGRF, Distillation
from kornia.feature.loftr.backbone.resnet_fpn import ResNetFPN_8_2


def main():
    torch.set_num_threads(4)
    torch.manual_seed(2089)
    root = Path('/home/zhang/leader-image-gate-lpgrf')
    checkpoint = root/'assets/loftr_outdoor.ckpt'
    official = ResNetFPN_8_2(dict(initial_dim=128, block_dims=[128, 196, 256])).cuda().eval()
    state = torch.load(checkpoint, weights_only=True)['state_dict']
    official.load_state_dict({k[len('backbone.'):]:v for k, v in state.items() if k.startswith('backbone.')})
    local = LoFTRLocal().load_pretrained(checkpoint).cuda().eval()
    image = torch.rand(1, 1, 64, 80, device='cuda')
    with torch.no_grad():
        expected = official.layer3(official.layer2(official.layer1(official.relu(official.bn1(official.conv1(image))))))
        difference = float((expected-local(image)).abs().max())
    assert difference == 0
    gate = LPGRF().cuda()
    lidar = torch.randn(16, 512, device='cuda', requires_grad=True)
    visual = torch.randn(16, 128, device='cuda', requires_grad=True)
    invalid = torch.zeros(16, device='cuda', dtype=torch.bool)
    distance = torch.ones(16, device='cuda')
    assert torch.equal(gate(lidar, torch.full_like(visual, float('nan')), invalid, distance), lidar)
    distill = Distillation().cuda()
    loss = distill(lidar, visual, ~invalid, ~invalid)
    loss.backward()
    assert lidar.grad is None and visual.grad.norm() > 0
    assert torch.isfinite(distill(lidar, torch.full_like(visual, float('nan')), invalid, ~invalid))
    result = dict(official_loftr_layer3_max_error=difference, invalid_nonfinite_feature_fallback=True,
        distillation_teacher_stopgrad=True, distillation_visual_gradient=True,
        fusion_parameters=sum(p.numel() for p in gate.parameters()), distillation_parameters=sum(p.numel() for p in distill.parameters()))
    (root/'implementation_checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

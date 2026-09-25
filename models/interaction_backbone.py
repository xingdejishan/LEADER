import math

import MinkowskiEngine as ME
import torch
from torch import nn
from torch.nn import functional as F

from models.surface_token_fusion import surface_support


class GeometryImageExchange(nn.Module):
    def __init__(self, channels, hidden=64):
        super().__init__()
        self.geometry = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, hidden))
        self.image_update = nn.Sequential(nn.Conv2d(hidden*2, hidden, 3, padding=1),
                                          nn.GroupNorm(8, hidden), nn.GELU(), nn.Conv2d(hidden, hidden, 3, padding=1))
        self.query = nn.Linear(channels, hidden)
        self.key = nn.Conv2d(hidden, hidden, 1)
        self.value = nn.Conv2d(hidden, hidden, 1)
        self.output = nn.Linear(hidden, channels, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, sparse, image, inputs):
        features = sparse.F
        pixels, offsets, support = surface_support(inputs['points'], sparse.C, sparse.tensor_stride,
            inputs['intrinsics'], inputs['camera_from_lidar'], inputs['recovery'],
            inputs['image_bounds'], inputs.get('image_valid_mask'), .2, 1024)
        batch_index = sparse.C[:, 0].long().to(features.device)
        geometries = self.geometry(features)
        updated_images, visual = [], features.new_zeros((len(features), image.shape[1]))
        for batch in range(len(image)):
            rows = torch.where((batch_index == batch) & support.any(-1))[0]
            canvas = image.new_zeros((64*64, image.shape[1]))
            mass = image.new_zeros((64*64, 1))
            if len(rows):
                selected = pixels[rows]
                valid = support[rows]
                cells = torch.floor((selected+.5)/16).long().clamp(0, 63)
                flat = cells[..., 1]*64+cells[..., 0]
                source = geometries[rows, None].expand(-1, 8, -1)
                canvas.index_add_(0, flat[valid], source[valid])
                mass.index_add_(0, flat[valid], image.new_ones((int(valid.sum()), 1)))
            canvas = (canvas/mass.clamp_min(1)).T.reshape(1, image.shape[1], 64, 64)
            current = image[batch:batch+1]
            updated = current+self.image_update(torch.cat((current, canvas), 1))
            mask = inputs.get('image_valid_mask')
            if mask is not None:
                updated = updated*mask[batch:batch+1]
            updated_images.append(updated)
            if not len(rows):
                continue
            grid = ((pixels[rows]+.5)*(2/1024)-1)[None]
            keys = F.grid_sample(self.key(updated), grid, align_corners=False)[0].permute(1, 2, 0)
            values = F.grid_sample(self.value(updated), grid, align_corners=False)[0].permute(1, 2, 0)
            logits = (keys*self.query(features[rows])[:, None]).sum(-1)/math.sqrt(keys.shape[-1])
            weights = logits.masked_fill(~support[rows], -1e4).softmax(-1)
            visual[rows] = (weights[..., None]*values).sum(1)
        output = features+self.output(visual)
        return ME.SparseTensor(output, coordinate_map_key=sparse.coordinate_map_key,
                               coordinate_manager=sparse.coordinate_manager), torch.cat(updated_images)


class InterleavedInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Conv2d(256, 64, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.exchanges = nn.ModuleDict({'1': GeometryImageExchange(64), '3': GeometryImageExchange(256)})

    def encode(self, encoder, sparse, batch):
        inputs = {key: batch[key].to(sparse.F.device) for key in
                  ('intrinsics', 'camera_from_lidar', 'image_bounds', 'image_valid_mask') if key in batch}
        inputs['points'] = batch['points']
        count = len(batch['sam_features'])
        transform = batch.get('T_corr', torch.eye(4)[None].expand(count, -1, -1)).to(sparse.F.device)
        inputs['recovery'] = torch.linalg.inv(transform)
        state = [self.image_encoder(batch['sam_features'].to(sparse.F.device))]
        if 'image_valid_mask' in inputs:
            state[0] = state[0]*inputs['image_valid_mask']

        def exchange(stage, value):
            if str(stage) not in self.exchanges:
                return value
            value, state[0] = self.exchanges[str(stage)](value, state[0], inputs)
            return value

        return encoder(sparse, interaction=exchange)

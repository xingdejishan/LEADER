import torch

from lscr_v3_visual_residual import GRID, target_local_offsets


def test_target_local_offsets_keeps_batch_index():
    offsets = torch.zeros((2, 2, GRID, GRID))
    first_cell, second_cell = 7, 143
    first_y, first_x = divmod(first_cell, GRID)
    second_y, second_x = divmod(second_cell, GRID)
    offsets[0, :, first_y, first_x] = torch.tensor((1., 2.))
    offsets[1, :, second_y, second_x] = torch.tensor((3., 4.))

    selected = target_local_offsets(offsets, torch.tensor((first_cell, second_cell)))

    assert torch.equal(selected, torch.tensor(((1., 2.), (3., 4.))))

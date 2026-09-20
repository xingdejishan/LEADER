import numpy as np

from xrefine_adapter import full_patch_valid, project_to_xrefine, xrefine_to_project


def test_xrefine_coordinate_round_trip():
    pixels = np.array(((0., 1.), (12.25, 7.75)))
    assert np.array_equal(xrefine_to_project(project_to_xrefine(pixels)), pixels)


def test_full_patch_valid_rejects_boundary_and_hole():
    mask = np.ones((20, 20), dtype=bool)
    mask[10, 10] = False
    valid, _ = full_patch_valid(mask, np.array(((10., 10.), (5., 5.), (1., 1.))))
    assert np.array_equal(valid, np.array((False, False, False)))

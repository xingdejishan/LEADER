import numpy as np

from research.prevoxel_multiview.g1_irls_staged_cache import selected_rows


def test_selected_rows_keeps_validation_order_and_limit():
    rows = [{"frame_id": "train", "split": "train"}, {"frame_id": "one", "split": "val"},
            {"frame_id": "two", "split": "validation"}, {"frame_id": "test", "split": "test"}]
    assert [row["frame_id"] for row in selected_rows(rows, "validation", 2)] == ["one", "two"]


def test_selected_rows_selects_only_train_for_train_split():
    rows = [{"frame_id": "one", "split": "train"}, {"frame_id": "two", "split": "val"}]
    assert np.array_equal([row["frame_id"] for row in selected_rows(rows, "train", 0)], ["one"])

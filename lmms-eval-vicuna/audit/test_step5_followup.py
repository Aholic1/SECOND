#!/usr/bin/env python3
"""Small CPU-only checks for the Step 5 follow-up helpers; no model inference."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit"))

import step5_unstable_repeatability as repeatability  # noqa: E402
import step5_visualize_mechanism_candidates as visualization  # noqa: E402


class FollowupHelperTests(unittest.TestCase):
    def test_targeted_repeat_set_and_schedule_are_frozen(self):
        self.assertEqual(repeatability.QUESTION_IDS, ("457", "1512"))
        self.assertEqual(repeatability.ADDITIONAL_LABELS, ("C3", "D3", "C4", "D4", "C5", "D5"))

    def test_nearest_resize_matches_runtime_floor_indexing(self):
        source = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        actual = visualization.nearest_resize(source, (4, 6))
        expected = np.asarray([
            [1, 1, 2, 2, 3, 3],
            [1, 1, 2, 2, 3, 3],
            [4, 4, 5, 5, 6, 6],
            [4, 4, 5, 5, 6, 6],
        ], dtype=np.int32)
        np.testing.assert_array_equal(actual, expected)

    def test_touching_box_edges_do_not_count_as_intersection(self):
        self.assertFalse(visualization.rect_intersects([0, 0, 5, 5], [5, 0, 8, 5]))
        self.assertTrue(visualization.rect_intersects([0, 0, 5, 5], [4, 4, 8, 8]))

    def test_coco_xywh_boxes_are_converted_to_xyxy(self):
        self.assertEqual(
            visualization.target_rectangles({"target_bboxes": [{"bbox_xywh": [2, 3, 5, 7]}]}),
            [[2.0, 3.0, 7.0, 10.0]],
        )


if __name__ == "__main__":
    unittest.main()

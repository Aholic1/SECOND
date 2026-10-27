import unittest
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step6_causal_v3 import map_consumed_edits


class EditTests(unittest.TestCase):
    def test_same_budget_swap(self):
        applied = np.zeros((48, 48), dtype=np.float32)
        applied[0, 0] = 1
        applied[0, 2] = 1
        consumed = np.zeros((24, 24), dtype=np.float32)
        consumed[0, 0] = 1
        consumed[0, 1] = 1
        result = map_consumed_edits(applied, consumed, [2], [0])
        self.assertEqual(int(result.sum()), 2)

    def test_remove_and_add_are_disjoint(self):
        applied = np.ones((48, 48), dtype=np.float32)
        consumed = np.ones((48, 48), dtype=np.float32)
        applied[0, 0] = 0
        consumed[0, 0] = 0
        result = map_consumed_edits(applied, consumed, [0], [1])
        self.assertEqual(int(result.sum()), 48 * 48 - 1)
        self.assertEqual(result[0, 0], 1)
        self.assertEqual(result[0, 1], 0)


if __name__ == '__main__':
    unittest.main()

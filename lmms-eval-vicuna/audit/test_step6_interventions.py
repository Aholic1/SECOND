import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step6_interventions import lift_additions
from step5_visualize_mechanism_candidates import nearest_resize


class MappingTests(unittest.TestCase):
    def test_actual_consumed_budget_each_stage(self):
        for size in (12, 24, 48):
            original = np.zeros((48, 48), dtype=np.float32)
            consumed = nearest_resize(original, (size, size))
            result = lift_additions(original, consumed, [0, size + 1, size * size - 1])
            observed = nearest_resize(result, (size, size))
            self.assertEqual(int(observed.sum()), 3)
            self.assertEqual(np.flatnonzero(observed).tolist(), [0, size + 1, size * size - 1])
            self.assertEqual(int(original.sum()), 0)

    def test_kept_cells_remain_kept(self):
        original = np.ones((48, 48), dtype=np.float32)
        original[0, 0] = 0
        consumed = nearest_resize(original, (12, 12))
        result = lift_additions(original, consumed, [0])
        self.assertEqual(int(nearest_resize(result, (12, 12)).sum()), 144)

    def test_disabled_hook_returns_identity_without_torch(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "lmms_eval/second_intervention.py"
        spec = importlib.util.spec_from_file_location("intervention_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = object()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIs(module.intervene_mask(value, "1", 0, 1, "D"), value)


if __name__ == "__main__":
    unittest.main()

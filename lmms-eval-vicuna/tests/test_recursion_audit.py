import math
import unittest

import torch

from lmms_eval.recursion_utils import attn_entropy_topk_based_recursion


class RecursionAuditTests(unittest.TestCase):
    def test_diagnostics_preserve_legacy_mask_and_keep_ratio(self):
        attention = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=torch.float32
        )

        legacy_mask, legacy_keep = attn_entropy_topk_based_recursion(
            attention, torch.zeros_like(attention), 1.0
        )
        mask, keep, diagnostics = attn_entropy_topk_based_recursion(
            attention,
            torch.zeros_like(attention),
            1.0,
            return_diagnostics=True,
        )

        self.assertTrue(torch.equal(mask, legacy_mask))
        self.assertEqual(keep, legacy_keep)
        self.assertEqual(diagnostics["selected_token_count"], int(mask.sum().item()))
        self.assertEqual(diagnostics["total_token_count"], attention.numel())
        self.assertEqual(diagnostics["selection_grid_shape"], [2, 4])
        self.assertAlmostEqual(diagnostics["requested_keep_ratio"], keep)
        self.assertTrue(math.isfinite(diagnostics["entropy_2d"]))

    def test_ties_distinguish_requested_and_actual_keep_ratio(self):
        attention = torch.tensor([[4.0, 1.0, 1.0, 1.0]] * 4)

        _, requested, diagnostics = attn_entropy_topk_based_recursion(
            attention,
            torch.zeros_like(attention),
            0.3,
            return_diagnostics=True,
        )

        self.assertGreater(diagnostics["actual_keep_ratio"], requested)
        self.assertEqual(diagnostics["actual_keep_ratio"], 1.0)

    def test_non_finite_attention_is_rejected(self):
        attention = torch.tensor([[1.0, float("nan")], [2.0, 3.0]])

        with self.assertRaisesRegex(ValueError, "finite"):
            attn_entropy_topk_based_recursion(
                attention,
                torch.zeros_like(attention),
                1.0,
                return_diagnostics=True,
            )


if __name__ == "__main__":
    unittest.main()

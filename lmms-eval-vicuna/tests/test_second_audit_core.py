from pathlib import Path
import tempfile
import unittest

import torch

from lmms_eval.second_audit import (
    applied_selection_mask,
    applies_selection,
    binary_logit_record,
    parse_audit_artifact_mode,
    resolve_binary_token_ids,
    select_final_logits,
    uses_contrastive_decoding,
    uses_recursion,
    validate_audit_args,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            "prompt": [10, 11],
            "Yes": [2],
            "No": [4],
        }
        return mapping[text]


class SecondAuditCoreTests(unittest.TestCase):
    def test_raw_logits_are_exact_last_stage_logits(self):
        stages = [torch.tensor([[1.0, 3.0]]), torch.tensor([[2.0, 2.5]])]

        actual = select_final_logits(stages, [0.8], use_cd=False)

        self.assertTrue(torch.equal(actual, stages[-1]))
        self.assertIsNot(actual, stages[-1])

    def test_cd_logits_match_existing_stage_difference_formula(self):
        stages = [
            torch.tensor([[1.0, 3.0, 0.0]]),
            torch.tensor([[2.0, 2.5, 1.0]]),
        ]

        actual = select_final_logits(stages, [0.5], use_cd=True)
        expected = stages[-1] + 0.5 * (stages[-1] - stages[0])

        self.assertTrue(torch.equal(actual, expected))

    def test_condition_c_and_d_share_cd_but_only_d_applies_selection(self):
        self.assertTrue(uses_recursion("B"))
        self.assertFalse(uses_recursion("A"))
        self.assertTrue(uses_contrastive_decoding("C"))
        self.assertTrue(uses_contrastive_decoding("D"))
        self.assertFalse(uses_contrastive_decoding("B"))
        self.assertFalse(applies_selection("C"))
        self.assertTrue(applies_selection("D"))

    def test_alltoken_applies_ones_without_mutating_proposal(self):
        proposed = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        original = proposed.clone()

        applied = applied_selection_mask("C", proposed)

        self.assertTrue(torch.equal(proposed, original))
        self.assertTrue(torch.equal(applied, torch.ones_like(proposed)))

    def test_audit_requires_output_directory(self):
        with self.assertRaisesRegex(ValueError, "audit_output_dir"):
            validate_audit_args("D", None, 0)

    def test_artifact_mode_is_explicit(self):
        self.assertEqual(parse_audit_artifact_mode("VISUAL").value, "visual")
        self.assertEqual(parse_audit_artifact_mode("compact").value, "compact")
        with self.assertRaisesRegex(ValueError, "audit_artifact_mode"):
            parse_audit_artifact_mode("all")

    def test_binary_margin_uses_generated_label_tokens(self):
        token_ids = resolve_binary_token_ids(FakeTokenizer(), "prompt")
        logits = torch.tensor([[0.0, 0.0, 3.0, 0.0, 1.5]])

        record = binary_logit_record(logits, token_ids)

        self.assertTrue(record["available"])
        self.assertEqual(record["yes_token_id"], 2)
        self.assertEqual(record["no_token_id"], 4)
        self.assertEqual(record["yes_no_margin"], 1.5)

    def test_binary_ids_use_the_actual_standalone_generation_tokens(self):
        class BoundarySensitiveTokenizer:
            def encode(self, text, add_special_tokens=False):
                return {
                    "Yes": [3869],
                    "No": [1939],
                    "promptYes": [10, 8241],
                    "promptNo": [10, 3782],
                }[text]

        token_ids = resolve_binary_token_ids(BoundarySensitiveTokenizer(), "prompt")

        self.assertEqual(token_ids["yes_token_id"], 3869)
        self.assertEqual(token_ids["no_token_id"], 1939)


if __name__ == "__main__":
    unittest.main()

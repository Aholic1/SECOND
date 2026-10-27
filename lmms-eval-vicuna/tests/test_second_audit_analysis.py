import unittest

from tools.second_audit import analyze_pairs, verify_prediction_parity, wilson_interval


def paired_row(question_id, stratum, a, b, c, d):
    return {
        "question_id": question_id,
        "stratum": stratum,
        "A": {"correct": a, "prediction": "yes" if a else "no"},
        "B": {"correct": b, "prediction": "yes" if b else "no"},
        "C": {"correct": c, "prediction": "yes" if c else "no"},
        "D": {"correct": d, "prediction": "yes" if d else "no"},
    }


class SecondAuditAnalysisTests(unittest.TestCase):
    def test_paired_analysis_uses_c_to_d_for_selection_harm(self):
        rows = [
            paired_row("q1", "FN", True, True, True, False),
            paired_row("q2", "FP", False, False, False, True),
            paired_row("q3", "TP", True, True, True, True),
        ]

        report = analyze_pairs(rows)

        self.assertEqual(report["selection"]["harm_count"], 1)
        self.assertEqual(report["selection"]["benefit_count"], 1)
        self.assertEqual(report["fn_rescue"]["count"], 1)
        self.assertEqual(report["fn_rescue"]["denominator"], 1)

    def test_b_to_c_is_reported_as_cd_not_selection(self):
        rows = [paired_row("q1", "TP", True, True, False, False)]

        report = analyze_pairs(rows)

        self.assertEqual(report["contrastive_decoding"]["harm_count"], 1)
        self.assertEqual(report["selection"]["harm_count"], 0)

    def test_parity_rejects_any_d_prediction_change(self):
        with self.assertRaisesRegex(ValueError, "prediction parity"):
            verify_prediction_parity(
                original={"q1": "yes", "q2": "no"},
                new={"q1": "no", "q2": "no"},
            )

    def test_parity_rejects_missing_ids(self):
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            verify_prediction_parity(original={"q1": "yes"}, new={"q2": "yes"})

    def test_wilson_interval_is_bounded(self):
        lower, upper = wilson_interval(5, 10)

        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)


if __name__ == "__main__":
    unittest.main()

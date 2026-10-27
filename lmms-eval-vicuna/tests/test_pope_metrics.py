import importlib.util
from pathlib import Path
import unittest


UTILS_PATH = Path(__file__).parents[1] / "lmms_eval" / "tasks" / "pope" / "utils.py"
SPEC = importlib.util.spec_from_file_location("pope_utils_under_test", UTILS_PATH)
utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utils)


def make_result(prediction, ground_truth):
    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "score": float(prediction == ground_truth),
    }


class PopeMetricTests(unittest.TestCase):
    def test_explicit_yes_ratios_and_invalid_ratio(self):
        rows = [
            make_result("yes", "yes"),
            make_result("no", "yes"),
            make_result("invalid", "no"),
        ]

        self.assertAlmostEqual(utils.pope_aggregate_pred_yes_ratio(rows), 1 / 3)
        self.assertAlmostEqual(utils.pope_aggregate_gt_yes_ratio(rows), 2 / 3)
        self.assertAlmostEqual(utils.pope_aggregate_invalid_ratio(rows), 1 / 3)
        self.assertAlmostEqual(utils.pope_aggregate_yes_ratio(rows), 1 / 3)

    def test_process_results_normalizes_first_binary_word(self):
        metrics = utils.pope_process_results(
            {"question_id": "q", "answer": "yes"}, ["Yes."]
        )

        self.assertEqual(metrics["pope_accuracy"]["prediction"], "yes")
        self.assertIn("pope_pred_yes_ratio", metrics)
        self.assertIn("pope_gt_yes_ratio", metrics)
        self.assertIn("pope_invalid_ratio", metrics)

    def test_invalid_positive_prediction_counts_as_recall_miss(self):
        rows = [make_result("invalid", "yes"), make_result("yes", "yes")]

        self.assertEqual(utils.pope_aggregate_recall(rows), 0.5)


if __name__ == "__main__":
    unittest.main()

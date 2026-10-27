#!/usr/bin/env python3
"""Test script for second_audit.py new commands."""

import json
import subprocess
import sys
from pathlib import Path
import tempfile


def run_command(cmd, check=True):
    """Run a command and return stdout."""
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=check)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        if check:
            sys.exit(1)
    return result.stdout


def test_index_command():
    """Test the index command with a small sample."""
    print("\n=== Testing 'index' command ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a minimal test input
        test_input = tmpdir / "test_samples.jsonl"
        samples = [
            {
                "doc": {"question_id": "1", "id": "100", "question": "Is there a cat?", "answer": "yes"},
                "pope_accuracy": {"question_id": "1", "prediction": "yes", "ground_truth": "yes"}
            },
            {
                "doc": {"question_id": "2", "id": "200", "question": "Is there a dog?", "answer": "no"},
                "pope_accuracy": {"question_id": "2", "prediction": "no", "ground_truth": "no"}
            },
            {
                "doc": {"question_id": "3", "id": "300", "question": "Is there a bird?", "answer": "yes"},
                "pope_accuracy": {"question_id": "3", "prediction": "no", "ground_truth": "yes"}
            }
        ]

        with test_input.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        output = tmpdir / "index.jsonl"

        run_command([
            "python3", "tools/second_audit.py", "index",
            "--samples-jsonl", str(test_input),
            "--output", str(output)
        ])

        # Verify output
        records = []
        with output.open() as f:
            for line in f:
                records.append(json.loads(line))

        assert len(records) == 3, f"Expected 3 records, got {len(records)}"
        assert records[0]["stratum"] == "TP"
        assert records[1]["stratum"] == "TN"
        assert records[2]["stratum"] == "FN"

        print("✓ index command passed")


def test_summarize_command():
    """Test the summarize command."""
    print("\n=== Testing 'summarize' command ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test index
        test_index = tmpdir / "index.jsonl"
        records = [
            {"question_id": "1", "stratum": "TP", "gt": "yes", "pred": "yes", "correct": True},
            {"question_id": "2", "stratum": "TN", "gt": "no", "pred": "no", "correct": True},
            {"question_id": "3", "stratum": "FN", "gt": "yes", "pred": "no", "correct": False},
            {"question_id": "4", "stratum": "FP", "gt": "no", "pred": "yes", "correct": False},
        ]

        with test_index.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        output = tmpdir / "summary.json"

        run_command([
            "python3", "tools/second_audit.py", "summarize",
            "--index", str(test_index),
            "--output", str(output)
        ])

        # Verify output
        summary = json.loads(output.read_text())

        assert summary["total"] == 4
        assert summary["counts"]["TP"] == 1
        assert summary["counts"]["TN"] == 1
        assert summary["counts"]["FN"] == 1
        assert summary["counts"]["FP"] == 1
        assert summary["accuracy"] == 0.5

        print("✓ summarize command passed")


def test_extract_features_command():
    """Test the extract_features command."""
    print("\n=== Testing 'extract_features' command ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test index
        test_index = tmpdir / "index.jsonl"
        records = [
            {"question_id": "1", "stratum": "TP", "gt": "yes", "pred": "yes", "correct": True,
             "question": "Is there a cat next to a dog in the image?", "image_id": "100"},
            {"question_id": "2", "stratum": "FN", "gt": "yes", "pred": "no", "correct": False,
             "question": "Is there a bird in the image?", "image_id": "200"},
        ]

        with test_index.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        # Create test image mapping
        test_mapping = tmpdir / "mapping.jsonl"
        mappings = [
            {"question_id": "1", "width": 640, "height": 480, "readable": True},
            {"question_id": "2", "width": 800, "height": 600, "readable": True},
        ]

        with test_mapping.open("w") as f:
            for mapping in mappings:
                f.write(json.dumps(mapping) + "\n")

        output = tmpdir / "features.jsonl"

        run_command([
            "python3", "tools/second_audit.py", "extract_features",
            "--index", str(test_index),
            "--image-mapping", str(test_mapping),
            "--output", str(output)
        ])

        # Verify output
        features = []
        with output.open() as f:
            for line in f:
                features.append(json.loads(line))

        assert len(features) == 2
        assert features[0]["has_relation"] == True  # "next to" detected
        assert features[1]["has_relation"] == False
        assert features[0]["target_object"] == "cat next to a dog"

        print("✓ extract_features command passed")


def main():
    print("========================================")
    print("Testing second_audit.py new commands")
    print("========================================")

    try:
        test_index_command()
        test_summarize_command()
        test_extract_features_command()

        print("\n========================================")
        print("All tests passed! ✓")
        print("========================================")
        print("\nYou can now run the full Step 1 pipeline:")
        print("  bash audit/step1_run.sh")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

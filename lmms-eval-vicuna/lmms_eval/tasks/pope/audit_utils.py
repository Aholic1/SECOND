import json
import os
from pathlib import Path


def _load_question_ids(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {1, 2}:
        raise ValueError(f"unsupported audit manifest schema: {payload.get('schema_version')}")
    question_ids = [str(row.get("question_id", "")) for row in payload.get("samples", [])]
    if not question_ids or any(not value for value in question_ids):
        raise ValueError("audit manifest contains an empty question_id set")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("audit manifest contains duplicate question_id values")
    return set(question_ids)


def pope_process_audit_docs(dataset):
    manifest_path = os.environ.get("SECOND_AUDIT_MANIFEST")
    if not manifest_path:
        raise ValueError("SECOND_AUDIT_MANIFEST is required for pope_pop_audit")

    wanted = _load_question_ids(manifest_path)
    filtered = dataset.filter(lambda doc: str(doc["question_id"]) in wanted)
    found = {str(doc["question_id"]) for doc in filtered}
    missing = sorted(wanted - found)
    unexpected = sorted(found - wanted)
    if missing or unexpected or len(filtered) != len(wanted):
        raise ValueError(
            "audit manifest/dataset mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"expected_count={len(wanted)}, actual_count={len(filtered)}"
        )
    return filtered

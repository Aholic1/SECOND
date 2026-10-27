import re


_BINARY_ANSWER_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def normalize_pope_answer(text):
    match = _BINARY_ANSWER_RE.match(str(text))
    return match.group(1).lower() if match else "invalid"


def pope_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def pope_doc_to_text(doc):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{question}\nAnswer the question using a single word or phrase."


def pope_process_results(doc, results):
    pred = normalize_pope_answer(results[0])
    gt_ans = doc["answer"].lower().strip()
    assert gt_ans in ["yes", "no"]
    score = 1.0 if pred == gt_ans else 0.0
    return {
        "pope_accuracy": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_precision": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_recall": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_f1_score": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_yes_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_pred_yes_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_gt_yes_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
        "pope_invalid_ratio": {"question_id": doc["question_id"], "score": score, "prediction": pred, "ground_truth": gt_ans},
    }


def pope_aggregate_accuracy(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score


def pope_aggregate_precision(results):
    true_positives = 0
    false_positives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "no" and pred == "yes":
            false_positives += 1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def pope_aggregate_recall(results):
    true_positives = 0
    false_negatives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["ground_truth"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "yes" and pred != "yes":
            false_negatives += 1
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def pope_aggregate_f1_score(results):
    precision = pope_aggregate_precision(results)
    recall = pope_aggregate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def pope_aggregate_yes_ratio(results):
    return pope_aggregate_pred_yes_ratio(results)


def pope_aggregate_pred_yes_ratio(results):
    if not results:
        return 0
    return sum(result["prediction"] == "yes" for result in results) / len(results)


def pope_aggregate_gt_yes_ratio(results):
    if not results:
        return 0
    return sum(result["ground_truth"] == "yes" for result in results) / len(results)


def pope_aggregate_invalid_ratio(results):
    if not results:
        return 0
    return sum(result["prediction"] == "invalid" for result in results) / len(results)

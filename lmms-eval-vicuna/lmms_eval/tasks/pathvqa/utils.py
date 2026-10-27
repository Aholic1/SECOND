import datetime
import json
import os
from collections import defaultdict

from loguru import logger as eval_logger

def pathvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def pathvqa_doc_to_text(doc):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{question}\nAnswer the question using a single word or phrase."

def pathvqa_process_results(doc, results):
    pred = results[0].lower().strip()
    gt_ans = doc["answer"].lower().strip()   
    category = "pathvqa_closed_accuracy" if gt_ans in ["yes", "no"] else "pathvqa_open_accuracy"
    if category == "pathvqa_closed_accuracy":        
        score = 1.0 if pred == gt_ans else 0.0
    else:
        score = 1.0 if gt_ans in pred else 0.0
    return {
        category: {"category": category, "score": score, "prediction": pred, "ground_truth": gt_ans},        
    }

def pathvqa_aggregate_results(results):
    category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        category = result["category"]
        category_scores[category].append(score)

    category_avg_score = {}
    for category, scores in category_scores.items():
        avg_score = sum(scores) / len(scores)
        category_avg_score[category] = avg_score
        eval_logger.info(f"{category}: {avg_score:.2f}")

    avg_score = sum(category_avg_score.values()) / len(category_avg_score)
    return avg_score
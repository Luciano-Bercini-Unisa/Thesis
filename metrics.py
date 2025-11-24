"""
Compute evaluation metrics for LLM-based smart-contract vulnerability detection.

This script loads the raw per-contract predictions produced by measure_multi_test.py,
compares them against the SmartBugs-Curated ground truth (via expected_map),
and derives per-class TP/FP/TN/FN, precision, recall, specificity, and F1-score.
It also aggregates results across multiple runs of the same prompt variant and
writes the averaged metrics to results/<prompt>/average_results.json.

Input:
    results/<prompt>/*_output.json   (raw model predictions)

Output:
    results/<prompt>/average_results.json
"""

import json
import glob
import statistics
from collections import defaultdict
from ground_truth_extraction import extract_ground_truth
from vulnerabilities_constants import  KEYS

def compute_metrics(gt, predictions):
    """
    gt: dict[file_name] -> dict[vuln_name:0/1].
    predictions: list of entries from measure_multi_test.
    """
    metrics = {k: {"tp":0,"fp":0,"tn":0,"fn":0} for k in KEYS}
    for item in predictions:
        fn = item["file_name"]
        pred = item["prediction_map"]
        if fn not in gt:
            continue
        exp = gt[fn]
        for cls in KEYS:
            real = exp[cls]
            llm = pred.get(cls, 0)
            if real == 1 and llm == 1:
                metrics[cls]["tp"] += 1
            elif real == 0 and llm == 1:
                metrics[cls]["fp"] += 1
            elif real == 0 and llm == 0:
                metrics[cls]["tn"] += 1
            elif real == 1 and llm == 0:
                metrics[cls]["fn"] += 1
    # Compute derived metrics.
    results = []
    for cls, m in metrics.items():
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "class": cls,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        })

    macro = {
        "precision": round(sum(r["precision"] for r in results) / len(results), 4),
        "recall": round(sum(r["recall"] for r in results) / len(results), 4),
        "specificity": round(sum(r["specificity"] for r in results) / len(results), 4),
        "f1_score": round(sum(r["f1_score"] for r in results) / len(results), 4),
    }

    return {"per_class": results, "macro_avg": macro}


def avg_metrics_rp(folder):
    gt = extract_ground_truth()

    files = sorted(glob.glob(f"./results/{folder}/*_output.json"))
    per_class_acc = defaultdict(lambda: defaultdict(list))
    macro_acc = defaultdict(list)
    for fpath in files:
        with open(fpath, "r") as f:
            run = json.load(f)
        r = compute_metrics(gt, run)
        for cls_entry in r["per_class"]:
            c = cls_entry["class"]
            for key, val in cls_entry.items():
                if key != "class":
                    per_class_acc[c][key].append(val)
        for k, v in r["macro_avg"].items():
            macro_acc[k].append(v)

        # build final average
    out = {"per_class": [], "macro_avg": {}}

    for cls, metrics in per_class_acc.items():
        avg_entry = {"class": cls}
        for key, arr in metrics.items():
            avg_entry[key] = round(sum(arr) / len(arr), 4)
            if key == "f1_score" and len(arr) > 1:
                avg_entry["f1_score_std"] = round(statistics.stdev(arr), 4)
        out["per_class"].append(avg_entry)

    out["macro_avg"] = {
        k: round(sum(v) / len(v), 4) for k, v in macro_acc.items()
    }

    with open(f"./results/{folder}/average_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    avg_metrics_rp(args.folder)

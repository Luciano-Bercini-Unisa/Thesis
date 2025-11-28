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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from ground_truth_extraction import extract_ground_truth
from vulnerabilities_constants import  CATEGORIES

def compute_metrics(gt, predictions):
    """
    gt: dict[file_name] -> dict[vuln_name:0/1].
    predictions: list of entries from measure_multi_test.
    """
    metrics = {k: {"tp":0,"fp":0,"tn":0,"fn":0} for k in CATEGORIES}
    for item in predictions:
        fn = item["file_name"]
        pred = item["prediction_map"]
        if fn not in gt:
            continue
        exp = gt[fn]
        for cls in CATEGORIES:
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
    # Build final average.
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
    # Combine all runs for confusion matrix.
    all_predictions = []
    for fpath in files:
        with open(fpath, "r") as f:
            run = json.load(f)
            all_predictions.extend(run)
    conf_mats = build_confusion_matrices(gt, all_predictions)
    out["confusion_matrices"] = {
        cls: conf_mats[cls].tolist()
        for cls in CATEGORIES
    }
    for cls, mat in conf_mats.items():
        save_confusion_matrix_png(mat, cls, folder)

    with open(f"./results/{folder}/average_results.json", "w") as f:
        json.dump(out, f, indent=2)


def build_confusion_matrices(gt, predictions):
    """
    Build per-class confusion matrices in the standard ML layout:

            Predicted
              +     -
    True +    TP    FN
    True -    FP    TN

    Matrix format returned:
         [TP, FN]
         [FP, TN]
    """
    matrices = {
        cls: np.zeros((2, 2), dtype=int)
        for cls in CATEGORIES
    }
    for item in predictions:
        fn = item["file_name"]
        if fn not in gt:
            continue
        true_map = gt[fn]
        pred_map = item["prediction_map"]
        for vulnerability in CATEGORIES:
            true = true_map[vulnerability]          # 0 or 1.
            pred = pred_map.get(vulnerability, 0)   # 0 or 1.
            if true == 1 and pred == 1:
                matrices[vulnerability][0, 0] += 1   # TP
            elif true == 1 and pred == 0:
                matrices[vulnerability][0, 1] += 1   # FN
            elif true == 0 and pred == 1:
                matrices[vulnerability][1, 0] += 1   # FP
            elif true == 0 and pred == 0:
                matrices[vulnerability][1, 1] += 1   # TN
    return matrices


def save_confusion_matrix_png(matrix, cls, folder):
    """
    matrix = 2x2 array:
    [TP, FN]
    [FP, TN]
    """
    tp, fn = matrix[0, 0], matrix[0, 1]
    fp, tn = matrix[1, 0], matrix[1, 1]
    mat = np.array([
        [tp, fn],
        [fp, tn]
    ])

    labels = np.array([
        [f"TP\n{tp}", f"FN\n{fn}"],
        [f"FP\n{fp}", f"TN\n{tn}"]
    ])

    plt.figure(figsize=(3.5, 3.2))
    sns.heatmap(
        matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred +", "Pred −"],
        yticklabels=["True +", "True −"]
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title(cls)
    plt.tight_layout()
    plt.savefig(f"./results/{folder}/{cls}_matrix.png", dpi=240)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    avg_metrics_rp(args.folder)

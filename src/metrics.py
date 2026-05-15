"""
Compute evaluation metrics for LLM-based smart-contract vulnerability detection.

This script loads the raw per-contract predictions produced by execution.py,
compares them against the SmartBugs-Curated ground truth (via expected_map),
and derives per-class TP/FP/TN/FN, precision, recall, specificity and F1-score.

For each run the script computes three averaging schemes:
- Macro:    arithmetic mean of the per-class metric (each class weighs 1/9).
- Micro:    pool TP/FP/FN across all classes, then compute a single P/R/F1.
            Dominated by frequent classes.
- Weighted: per-class metric averaged with weight equal to the class support
            (number of positives in ground truth).

All intermediate aggregations are computed and propagated as raw floats.
`compute_metrics()` returns raw floats by default; when called for storage
(e.g. when the JSON output is the final destination) pass `round_output=True`.
Cross-run aggregation in `avg_metrics_rp()` therefore uses raw floats, and
rounds to 4 decimal places only once, when writing average_results.json.

Input:
    results/<model>/<prompt>/run_*.json (raw model predictions)

Output:
    results/<model>/<prompt>/average_results.json.
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .ground_truth_extraction import extract_ground_truth
from .vulnerabilities_constants import CATEGORIES


# ---------------------------------------------------------------- helpers --

def _safe_div(a, b):
    return a / b if b > 0 else 0.0


def _f1(precision, recall):
    return _safe_div(2 * precision * recall, precision + recall)


def _round_floats(d, ndigits=4):
    """Round every float value in a dict; non-float values pass through."""
    return {k: (round(v, ndigits) if isinstance(v, float) else v) for k, v in d.items()}


# ---------------------------------------------------------------- metrics --

def compute_metrics(gt, predictions, round_output=False):
    """
    Compute per-class TP/FP/TN/FN from one run and derive macro, micro and
    weighted averages.

    gt:           dict[file_name] -> dict[vuln_name: 0/1].
    predictions:  list of entries from execution.py.
    round_output: if True, round all float fields to 4 decimal places before
                  returning. Use only when the returned dict is the final
                  destination; keep False when the result will be averaged
                  further (e.g. across runs).
    """
    counts = {k: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for k in CATEGORIES}
    for item in predictions:
        fn = item["file_name"]
        if fn not in gt:
            continue
        exp = gt[fn]
        pred = item["prediction_map"]
        for cls in CATEGORIES:
            real = exp[cls]
            llm = pred.get(cls, 0)
            if   real == 1 and llm == 1: counts[cls]["tp"] += 1
            elif real == 0 and llm == 1: counts[cls]["fp"] += 1
            elif real == 0 and llm == 0: counts[cls]["tn"] += 1
            elif real == 1 and llm == 0: counts[cls]["fn"] += 1

    # Per-class metrics in raw floating-point precision.
    raw_per_class = []
    for cls, m in counts.items():
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        p  = _safe_div(tp, tp + fp)
        r  = _safe_div(tp, tp + fn)
        sp = _safe_div(tn, tn + fp)
        f1 = _f1(p, r)
        raw_per_class.append({
            "class":       cls,
            "precision":   p,
            "recall":      r,
            "specificity": sp,
            "f1_score":    f1,
            "support":     tp + fn,           # Positives in the ground truth.
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    n = len(raw_per_class)

    # Macro: simple mean of per-class metrics (raw floats).
    macro_raw = {
        "precision":   sum(c["precision"]   for c in raw_per_class) / n,
        "recall":      sum(c["recall"]      for c in raw_per_class) / n,
        "specificity": sum(c["specificity"] for c in raw_per_class) / n,
        "f1_score":    sum(c["f1_score"]    for c in raw_per_class) / n,
    }

    # Micro: pool the confusion-matrix counts before deriving P/R/F1.
    tot_tp = sum(m["tp"] for m in counts.values())
    tot_fp = sum(m["fp"] for m in counts.values())
    tot_tn = sum(m["tn"] for m in counts.values())
    tot_fn = sum(m["fn"] for m in counts.values())
    micro_p  = _safe_div(tot_tp, tot_tp + tot_fp)
    micro_r  = _safe_div(tot_tp, tot_tp + tot_fn)
    micro_sp = _safe_div(tot_tn, tot_tn + tot_fp)
    micro_raw = {
        "precision":   micro_p,
        "recall":      micro_r,
        "specificity": micro_sp,
        "f1_score":    _f1(micro_p, micro_r),
    }

    # Weighted: per-class metric weighted by class support (raw floats).
    total_support = sum(c["support"] for c in raw_per_class)
    if total_support > 0:
        weighted_raw = {
            "precision":   sum(c["precision"]   * c["support"] for c in raw_per_class) / total_support,
            "recall":      sum(c["recall"]      * c["support"] for c in raw_per_class) / total_support,
            "specificity": sum(c["specificity"] * c["support"] for c in raw_per_class) / total_support,
            "f1_score":    sum(c["f1_score"]    * c["support"] for c in raw_per_class) / total_support,
        }
    else:
        weighted_raw = {"precision": 0.0, "recall": 0.0, "specificity": 0.0, "f1_score": 0.0}

    result = {
        "per_class":    raw_per_class,
        "macro_avg":    macro_raw,
        "micro_avg":    micro_raw,
        "weighted_avg": weighted_raw,
    }

    if round_output:
        result = {
            "per_class":    [_round_floats(c) for c in raw_per_class],
            "macro_avg":    _round_floats(macro_raw),
            "micro_avg":    _round_floats(micro_raw),
            "weighted_avg": _round_floats(weighted_raw),
        }

    return result


def avg_metrics_rp(model: str, prompt: str):
    """Average metrics across the multiple runs of one (model, prompt) folder."""
    results_dir = Path("results") / model / prompt
    gt = extract_ground_truth()

    files = sorted(results_dir.glob("run_*.json"))
    if not files:
        raise RuntimeError(f"No run_*.json files found in {results_dir}")

    per_class_acc = defaultdict(lambda: defaultdict(list))
    macro_acc    = defaultdict(list)
    micro_acc    = defaultdict(list)
    weighted_acc = defaultdict(list)

    for fpath in files:
        with open(fpath, "r") as f:
            run = json.load(f)
        # Raw floats — averaging happens before any rounding.
        r = compute_metrics(gt, run, round_output=False)

        for cls_entry in r["per_class"]:
            c = cls_entry["class"]
            for key, val in cls_entry.items():
                if key != "class":
                    per_class_acc[c][key].append(val)
        for k, v in r["macro_avg"].items():    macro_acc[k].append(v)
        for k, v in r["micro_avg"].items():    micro_acc[k].append(v)
        for k, v in r["weighted_avg"].items(): weighted_acc[k].append(v)

    # Per-class averages. Numerical fields rounded only here.
    out = {"per_class": []}
    for cls, metrics in per_class_acc.items():
        avg_entry = {"class": cls}
        for key, arr in metrics.items():
            avg_entry[key] = round(sum(arr) / len(arr), 4)
            if key == "f1_score" and len(arr) > 1:
                avg_entry["f1_score_std"] = round(statistics.stdev(arr), 4)
        out["per_class"].append(avg_entry)

    # Macro / micro / weighted averages.
    def _summarise(acc):
        summary = {
            "precision_mean":   round(sum(acc["precision"])   / len(acc["precision"]),   4),
            "recall_mean":      round(sum(acc["recall"])      / len(acc["recall"]),      4),
            "specificity_mean": round(sum(acc["specificity"]) / len(acc["specificity"]), 4),
            "f1_mean":          round(sum(acc["f1_score"])    / len(acc["f1_score"]),    4),
        }
        summary["f1_std"] = round(statistics.stdev(acc["f1_score"]), 4) \
            if len(acc["f1_score"]) > 1 else 0.0
        return summary

    out["macro_avg"]    = _summarise(macro_acc)
    out["micro_avg"]    = _summarise(micro_acc)
    out["weighted_avg"] = _summarise(weighted_acc)

    # Combine all predictions for confusion-matrix heatmaps.
    all_predictions = []
    for fpath in files:
        with open(fpath, "r") as f:
            all_predictions.extend(json.load(f))

    output_dir = results_dir / "confusion_matrix_heatmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    conf_mats = build_confusion_matrices(gt, all_predictions)
    for cls, mat in conf_mats.items():
        save_confusion_matrix_png(mat, cls, output_dir)

    with (results_dir / "average_results.json").open("w") as f:
        json.dump(out, f, indent=2)


# ----------------------------------------------------------- heatmaps --

def build_confusion_matrices(gt, predictions):
    matrices = {cls: np.zeros((2, 2), dtype=int) for cls in CATEGORIES}
    for item in predictions:
        fn = item["file_name"]
        if fn not in gt:
            continue
        true_map = gt[fn]
        pred_map = item["prediction_map"]
        for vuln in CATEGORIES:
            t = true_map[vuln]
            p = pred_map.get(vuln, 0)
            if   t == 1 and p == 1: matrices[vuln][0, 0] += 1
            elif t == 1 and p == 0: matrices[vuln][0, 1] += 1
            elif t == 0 and p == 1: matrices[vuln][1, 0] += 1
            elif t == 0 and p == 0: matrices[vuln][1, 1] += 1
    return matrices


def save_confusion_matrix_png(matrix, cls, folder: Path):
    tp, fn = matrix[0, 0], matrix[0, 1]
    fp, tn = matrix[1, 0], matrix[1, 1]
    labels = np.array([
        [f"TP\n{tp}", f"FN\n{fn}"],
        [f"FP\n{fp}", f"TN\n{tn}"],
    ])
    plt.figure(figsize=(3.5, 3.2))
    sns.heatmap(
        matrix, annot=labels, fmt="", cmap="Blues", cbar=False,
        xticklabels=["Pred +", "Pred -"], yticklabels=["True +", "True -"],
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title(cls)
    plt.tight_layout()
    plt.savefig(folder / f"{cls}.png", dpi=240)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    avg_metrics_rp(args.model, args.prompt)
"""
Evaluation script for LLM-based vulnerability detection results.

This script compares normalized binary outputs from the LLM against ground-truth
labels provided in `vulnerabilities.json`. It computes TP, FP, TN, FN per
vulnerability type and calculates Precision, Recall, Specificity, and F1 score.
"""

import os
import json, argparse
from typing import Dict, Any

# Mapping of vulnerability keys to display names
VUL_MAP = {
    "access_control": "Access Control",
    "arithmetic": "Arithmetic",
    "front_running": "Front Running",
    "reentrancy": "Reentrancy",
    "time_manipulation": "Time Manipulation",
    "unchecked_low_level_calls": "Unchecked Low Level Calls",
    "denial_of_service": "Denial Of Service",
    "bad_randomness": "Bad Randomness",
    "short_addresses": "Short Addresses",
}

# Initialize results storage for confusion matrix values
RES_MAP: Dict[str, Dict[str, int]] = {
    key: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for key in VUL_MAP
}


def normalize_string(input_string: str) -> str:
    """Lowercase string and remove spaces and dashes for robust matching."""
    return input_string.lower().replace(" ", "").replace("-", "")


def evaluate_file(findings: str, vulnerabilities: Any) -> None:
    """
    Compare LLM output in one result file with ground truth vulnerabilities.

    Args:
        filepath: Path to result file produced by LLM.
        vulnerabilities: Ground truth vulnerability annotations for the contract.
    """
    if not vulnerabilities:
        return
    vul_category = vulnerabilities[0]["category"]

    content = normalize_string(findings)

    for vul_key, vul_name in VUL_MAP.items():
        positive_str = normalize_string(f"{vul_name}:1")
        negative_str = normalize_string(f"{vul_name}:0")

        if vul_category == vul_key and positive_str in content:
            RES_MAP[vul_key]["TP"] += 1
        elif vul_category == vul_key and negative_str in content:
            RES_MAP[vul_key]["FN"] += 1
        elif vul_category != vul_key and positive_str in content:
            RES_MAP[vul_key]["FP"] += 1
        else:
            RES_MAP[vul_key]["TN"] += 1


def compute_metrics() -> Dict[str, Dict[str, float]]:
    """Compute Precision, Recall, Specificity, and F1 score for each vulnerability."""
    metrics = {
        key: {"Precision": 0.0, "Recall": 0.0, "Specificity": 0.0, "F1": 0.0}
        for key in VUL_MAP
    }
    results = []
    for vul_key, values in RES_MAP.items():
        tp, fp, tn, fn = values["TP"], values["FP"], values["TN"], values["FN"]

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        results.append({
            'class': vul_key,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'precision': round(float(precision), 2),
            'recall': round(float(recall), 2),
            'specificity': round(float(specificity), 2),
            'f1_score': round(float(f1), 2)
        })

        metrics[vul_key] = {
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity,
            "F1": f1,
        }
        # Macro average
    macro_precision = sum(c['precision'] for c in results) / len(results)
    macro_recall = sum(c['recall'] for c in results) / len(results)
    macro_f1 = sum(c['f1_score'] for c in results) / len(results)
    macro_specificity = sum(c['specificity'] for c in results) / len(results)

    return {
        'per_class': results,
        'macro_avg': {
            'precision': round(float(macro_precision), 2),
            'recall': round(float(macro_recall), 2),
            'specificity': round(float(macro_specificity), 2),
            'f1_score': round(float(macro_f1), 2)
        }
    }


def main(file: str, vuln_file: str = "../smartbugs-curated/vulnerabilities.json") -> None:
    filename = f"../results/{file}.json"
    """Main evaluation routine."""
    with open(vuln_file, "r", encoding="utf-8") as f:
        vulns = json.load(f)

    #apri file json 
    with open(filename, "r", encoding="utf-8") as f:
        results = json.load(f)

    for entry_r in results:
        for entry_v in vulns:
            if entry_v["name"] == entry_r["file_name"]:
                evaluate_file(entry_r["findings"], entry_v["vulnerabilities"])

    metrics = compute_metrics()
    with open(f"../results/{file}_metrics_rp.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metrics.")
    parser.add_argument('--filename', type=str, help='Filename to process', required=True)
    args = parser.parse_args()
    main(args.filename)


#python3 eval_RP.py --filename original/unlabelled_gpt-4/5_ORIGINAL_output    
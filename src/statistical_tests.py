"""
Statistical tests on key pairwise comparisons between prompt configurations.

For each comparison (configA vs configB) on the same model:
- Extracts the 5 per-run macro F1 values for both configurations from
  the run_*.json files produced by execution.py (at full precision,
  no rounding).
- Runs a two-sided Mann-Whitney U test via scipy.stats.
- Reports the rank-biserial correlation as effect size, in the
  range [-1, +1]:
    *  r = +1: all 5 values of configA are above all 5 of configB
    *  r =  0: distributions indistinguishable
    *  r = -1: vice versa

Comparisons are focused: 5 in total, chosen to cover the three core
claims of the thesis:
  (a) few-shot improves quality across all three main models;
  (b) role conditioning does not significantly affect quality;
  (c) chain-of-thought does not help (and may even hurt).

Usage (from the project root):
    python -m src.statistical_tests
"""

import json
import statistics
from pathlib import Path

from scipy.stats import mannwhitneyu

from .metrics import compute_metrics
from .ground_truth_extraction import extract_ground_truth


# Key comparisons to test.
COMPARISONS = [
    # Few-shot vs zero-shot (RQ1: few-shot improves quality).
    ("microsoft__phi-4",            "FS_SA",      "ZS_SA",    "Phi-4: FS vs ZS"),
    ("Qwen__Qwen2.5-7B-Instruct",   "FS_SA",      "ZS_SA",    "Qwen 7B: FS vs ZS"),
    ("Qwen__Qwen2.5-14B-Instruct",  "FS_SA",      "ZS_SA",    "Qwen 14B: FS vs ZS"),
    # Role vs base (RQ1: role does not improve quality).
    ("microsoft__phi-4",            "ZS_ROLE_SA", "ZS_SA",    "Phi-4: ROLE vs base"),
    ("Qwen__Qwen2.5-7B-Instruct",   "ZS_ROLE_SA", "ZS_SA",    "Qwen 7B: ROLE vs base"),
    ("Qwen__Qwen2.5-14B-Instruct",  "ZS_ROLE_SA", "ZS_SA",    "Qwen 14B: ROLE vs base"),
    # Chain-of-thought vs base (RQ1: CoT does not help).
    ("microsoft__phi-4",            "ZS_COT_SA",  "ZS_SA",    "Phi-4: CoT vs base"),
    ("Qwen__Qwen2.5-7B-Instruct",   "ZS_COT_SA",  "ZS_SA",    "Qwen 7B: CoT vs base"),
    ("Qwen__Qwen2.5-14B-Instruct",  "ZS_COT_SA",  "ZS_SA",    "Qwen 14B: CoT vs base"),
]


# Helpers.
def per_run_macro_f1(model: str, prompt: str) -> list[float]:
    """Macro F1 for each of the 5 runs (model, prompt), full precision."""
    gt = extract_ground_truth()
    folder = Path(f"results/{model}/{prompt}")
    values = []
    for run_file in sorted(folder.glob("run_*.json")):
        with run_file.open(encoding="utf-8") as f:
            predictions = json.load(f)
        result = compute_metrics(gt, predictions, round_output=False)
        values.append(result["macro_avg"]["f1_score"])
    return values


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Mann-Whitney U effect size, in [-1, +1]."""
    return 2.0 * u_stat / (n1 * n2) - 1.0


def run_comparison(model: str, config_a: str, config_b: str) -> dict:
    """Run a two-sided Mann-Whitney U test and compute the rank-biserial."""
    a = per_run_macro_f1(model, config_a)
    b = per_run_macro_f1(model, config_b)

    if len(a) == 0 or len(b) == 0:
        raise RuntimeError(
            f"Missing runs for {model}: configA={len(a)} runs, configB={len(b)} runs."
        )

    u_stat, p_value = mannwhitneyu(a, b, alternative="two-sided", method="exact")
    return {
        "mean_a":        statistics.mean(a),
        "mean_b":        statistics.mean(b),
        "n_a":           len(a),
        "n_b":           len(b),
        "delta_f1":      statistics.mean(a) - statistics.mean(b),
        "u_stat":        u_stat,
        "p_value":       p_value,
        "rank_biserial": rank_biserial(u_stat, len(a), len(b)),
        "values_a":      a,
        "values_b":      b,
    }


def main():
    header = (
        f"{'Comparison':<28} {'F1 a':>8} {'F1 b':>8} "
        f"{'delta':>9} {'p-value':>10} {'r':>8} {'sig.':>5}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for model, config_a, config_b, label in COMPARISONS:
        try:
            r = run_comparison(model, config_a, config_b)
        except Exception as e:
            print(f"{label:<28} ERROR: {e}")
            continue
        sig = "**" if r["p_value"] < 0.05 else "  "
        print(
            f"{label:<28} "
            f"{r['mean_a']:>8.4f} "
            f"{r['mean_b']:>8.4f} "
            f"{r['delta_f1']:>+9.4f} "
            f"{r['p_value']:>10.4f} "
            f"{r['rank_biserial']:>+8.3f} "
            f"{sig:>5}"
        )
    print(sep)
    print(
        "\nLegend:\n"
        "  delta = mean(configA) - mean(configB).\n"
        "  r = rank-biserial correlation in [-1, +1].\n"
        "  +1 means configA totally dominates, -1 means configB does,\n"
        "  0 means the distributions are indistinguishable.\n"
        "  **= p < 0.05\n"
        "  With n=5 per group, the smallest achievable p-value is 0.0079.\n"
    )


if __name__ == "__main__":
    main()
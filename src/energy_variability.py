"""
Analyse the across-run variability of per-execution energy and produce a
boxplot, to check whether any single anomalous run (outlier) skews the
reported means.

For each (model, prompt) folder it reads the per-run CSVs (run_XXX.csv),
computes the TOTAL energy of each run (sum of per-contract vd+sa energy),
and reports the distribution across runs. It flags outliers using the
standard 1.5*IQR rule and, separately, reports the median (which is robust
to outliers) alongside the mean.

It also draws one boxplot per model (six configurations each), saved as PNG.

Usage (same style as aggregation.py):
    python -m <package>.energy_variability
or, if run as a standalone file from the project root:
    python energy_variability.py

Outputs (all under results/energy_variability/):
    energy_variability_summary.csv   (one row per model/prompt)
    boxplot_<model>.png              (one boxplot figure per model)
"""

import csv
import statistics
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
# All outputs of this analysis go in a dedicated subfolder under results/.
OUTPUT_DIR = RESULTS_DIR / "energy_variability"
SUMMARY_CSV = OUTPUT_DIR / "energy_variability_summary.csv"

# Which phase to analyse: "vd", "sa", or "total" (vd+sa).
# The thesis reports per-execution energy on the combined pipeline by default,
# but the RQ2 table uses combined energy per run; set as you need.
PHASE = "total"   # "vd" | "sa" | "total"


def _per_run_energy(prompt_folder: Path):
    """
    Return a list with one TOTAL energy value per run for the chosen phase.
    Each run_XXX.csv is one run; its total energy is the sum over contracts.
    """
    values = []
    run_files = sorted(prompt_folder.glob("run_*.csv"))
    for c in run_files:
        d = pd.read_csv(c)
        for col in ["vd_energy_kwh", "sa_energy_kwh"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        vd = d["vd_energy_kwh"].sum() if "vd_energy_kwh" in d.columns else 0.0
        sa = d["sa_energy_kwh"].sum() if "sa_energy_kwh" in d.columns else 0.0
        if PHASE == "vd":
            values.append(vd)
        elif PHASE == "sa":
            values.append(sa)
        else:
            values.append(vd + sa)
    return values, [p.name for p in run_files]


def main():
    if not RESULTS_DIR.exists():
        raise SystemExit("results/ directory not found")

    # Create the dedicated output subfolder (results/energy_variability/).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    # Collect per-model data for the boxplots: {model: {prompt: [values]}}
    by_model = {}

    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            if not list(prompt_dir.glob("run_*.csv")):
                continue

            model = model_dir.name
            prompt = prompt_dir.name
            values, run_names = _per_run_energy(prompt_dir)
            if not values:
                continue

            n = len(values)
            mean = statistics.mean(values)
            median = statistics.median(values)
            std = statistics.stdev(values) if n >= 2 else 0.0
            lo, hi = min(values), max(values)
            cv = (std / mean * 100) if mean > 0 else 0.0
            spread = ((hi - lo) / mean * 100) if mean > 0 else 0.0

            print(f"\n{model} / {prompt}  (phase={PHASE}, {n} runs)")
            for name, v in zip(run_names, values):
                print(f"    {name}: {v:.6f}")
            print(f"    mean={mean:.6f}  median={median:.6f}  std={std:.6f}  "
                  f"CV={cv:.1f}%  min={lo:.6f}  max={hi:.6f}  range={spread:.1f}% of mean")

            summary_rows.append({
                "model": model,
                "prompt": prompt,
                "phase": PHASE,
                "n_runs": n,
                "mean_kwh": round(mean, 9),
                "median_kwh": round(median, 9),
                "std_kwh": round(std, 9),
                "cv_percent": round(cv, 2),
                "min_kwh": round(lo, 9),
                "max_kwh": round(hi, 9),
                "range_percent_of_mean": round(spread, 2),
            })

            by_model.setdefault(model, {})[prompt] = values

    # --- Write summary CSV. ---
    if summary_rows:
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSummary written to {SUMMARY_CSV} ({len(summary_rows)} rows).")

    # --- One boxplot per model. ---
    for model, prompts in by_model.items():
        labels = sorted(prompts.keys())
        data = [prompts[p] for p in labels]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_ylabel(f"Energy per run ({PHASE}) [kWh]")
        ax.set_title(f"Per-run energy distribution: {model}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        out = OUTPUT_DIR / f"boxplot_{model}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Boxplot saved: {out}")


if __name__ == "__main__":
    main()
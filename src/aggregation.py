"""
Build the consolidated CSV at results/final_results.csv, one row per
(model, prompt) combination.

For each (model, prompt) folder the script reads the corresponding
`average_results.json` produced by metrics.py and aggregates the raw CSV
files for energy / latency / token statistics.

Macro, micro and weighted averaged metrics are all included as columns.

If an `average_results.json` is missing the new aggregation sections
(micro_avg / weighted_avg), the script regenerates it from the run_*.json
files before reading.

Rounding policy
---------------
- Metric values are loaded from average_results.json, where they are already
  rounded to 4 decimals (single rounding point at write time).
- Aggregated columns (token averages, latencies, energy totals, per-1k
  energies, emissions) are rounded here at write time:
    * counts / latencies: 4 decimals
    * energy / emissions: 9 decimals (matches execution.py's CSV precision
      and avoids zeroing out small values).

Across-run variability
----------------------
- For each (model, prompt) the script also computes the standard deviation,
  minimum and maximum of the total energy for each run (VD, and VD+SA
  combined). This answers whether a single anomalous run (outlier) skews the
  reported mean. Each run_XXX.csv is one run; the per-run total is the sum of
  its per-contract energy. Std is the sample standard deviation (n-1).
"""

import json
import csv
import statistics
import pandas as pd
from pathlib import Path

from .metrics import avg_metrics_rp

RESULTS_DIR = Path("results")
OUT_CSV = RESULTS_DIR / "final_results.csv"

REQUIRED_AVG_KEYS = ("macro_avg", "micro_avg", "weighted_avg")


def _r(x, n_digits=4):
    """Round a numeric value to `n_digits` decimals, tolerating NaN/None."""
    try:
        return round(float(x), n_digits)
    except (TypeError, ValueError):
        return x


def _r_energy(x):
    """Higher precision rounding for energy/emission values that can be tiny."""
    return _r(x, 9)


def _spread(values, n_digits=9):
    """
    Return (std, min, max) across runs.
    std is the sample standard deviation (n-1); 0.0 if fewer than 2 runs.
    """
    vals = [float(v) for v in values if v is not None]
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 0.0
    return round(std, n_digits), round(lo, n_digits), round(hi, n_digits)


def _per_1k(energy_kwh: float, tokens: float) -> float:
    return energy_kwh / (tokens / 1000) if tokens > 0 else 0.0


def _avg_section(metrics_json: dict, section: str, prefix: str) -> dict:
    """Flatten one aggregation section into prefixed CSV columns."""
    s = metrics_json[section]
    return {
        f"{prefix}_precision_mean": s["precision_mean"],
        f"{prefix}_recall_mean": s["recall_mean"],
        f"{prefix}_specificity_mean": s["specificity_mean"],
        f"{prefix}_f1_mean": s["f1_mean"],
        f"{prefix}_f1_std": s["f1_std"],
    }


def _load_or_regenerate_metrics(model: str, prompt: str) -> dict:
    """Load average_results.json, regenerating if any aggregation is missing."""
    path = RESULTS_DIR / model / prompt / "average_results.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if all(k in data for k in REQUIRED_AVG_KEYS):
            return data
        print(f"  Regenerating stale {path} ...")
    else:
        print(f"  No average_results.json in {path.parent}, generating ...")
    avg_metrics_rp(model, prompt)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_raw_csvs(prompt_folder: Path):
    """
    Load all per-run CSVs. Returns:
      - df: concatenation of all runs (for the existing aggregate columns),
      - num_runs: number of run files,
      - per_run: dict of per-run TOTALS (one value per run) used to compute
                 across-run std / min / max.
    Each run_XXX.csv is a single run containing all contracts of that run.
    """
    csvs = sorted(prompt_folder.glob("run_*.csv"))
    dfs = []
    per_run = {
        "vd_energy_kwh": [],
        "sa_energy_kwh": [],
        "total_energy_kwh": [],
        "vd_latency_s": [],
        "combined_latency_s": [],
    }
    for c in csvs:
        d = pd.read_csv(c)
        for col in ["vd_energy_kwh", "sa_energy_kwh", "vd_latency_s", "sa_latency_s"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")
        vd_e = d["vd_energy_kwh"].sum()
        sa_e = d["sa_energy_kwh"].sum()
        vd_l = d["vd_latency_s"].sum()
        sa_l = d["sa_latency_s"].sum()
        per_run["vd_energy_kwh"].append(vd_e)
        per_run["sa_energy_kwh"].append(sa_e)
        per_run["total_energy_kwh"].append(vd_e + sa_e)
        per_run["vd_latency_s"].append(vd_l)
        per_run["combined_latency_s"].append(vd_l + sa_l)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True), len(csvs), per_run


def main():
    if not RESULTS_DIR.exists():
        raise SystemExit("results/ directory not found")

    rows = []

    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        for prompt_dir in sorted(model_dir.iterdir()):
            if not prompt_dir.is_dir():
                continue
            if not list(prompt_dir.glob("run_*.json")):
                continue

            model_name = model_dir.name
            prompt_name = prompt_dir.name
            print(f"Processing {model_name}/{prompt_name}")

            metrics_json = _load_or_regenerate_metrics(model_name, prompt_name)
            df, num_runs, per_run = _load_raw_csvs(prompt_dir)

            for col in [
                "vd_energy_kwh", "vd_emissions_kg", "vd_latency_s",
                "vd_input_tokens", "vd_output_tokens", "vd_total_tokens",
                "sa_energy_kwh", "sa_emissions_kg", "sa_latency_s",
                "sa_input_tokens", "sa_output_tokens", "sa_total_tokens",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            num_contracts = df[["category", "file_name"]].drop_duplicates().shape[0]

            vd_total_energy = df["vd_energy_kwh"].sum()
            sa_total_energy = df["sa_energy_kwh"].sum()
            vd_total_emissions = df["vd_emissions_kg"].sum()
            sa_total_emissions = df["sa_emissions_kg"].sum()

            vd_input_tokens = df["vd_input_tokens"].sum()
            vd_output_tokens = df["vd_output_tokens"].sum()
            vd_total_tokens = df["vd_total_tokens"].sum()
            sa_input_tokens = df["sa_input_tokens"].sum()
            sa_output_tokens = df["sa_output_tokens"].sum()
            sa_total_tokens = df["sa_total_tokens"].sum()

            vd_total_latency = df["vd_latency_s"].sum()
            sa_total_latency = df["sa_latency_s"].sum()

            combined_energy = vd_total_energy + sa_total_energy
            combined_emissions = vd_total_emissions + sa_total_emissions
            combined_latency = vd_total_latency + sa_total_latency

            single_execution_energy = (combined_energy / num_runs) if (num_runs and combined_energy > 0) else 0.0

            # Across-run variability (per the outlier question).
            vd_e_std, vd_e_min, vd_e_max = _spread(per_run["vd_energy_kwh"])
            tot_e_std, tot_e_min, tot_e_max = _spread(per_run["total_energy_kwh"])
            # Per-execution energy std == std of per-run total energy.
            per_run_e_std = tot_e_std

            row = {
                "model": model_name,
                "prompt": prompt_name,

                # --- Quality metrics (already rounded inside the JSON). ---
                **_avg_section(metrics_json, "macro_avg", "macro"),
                **_avg_section(metrics_json, "micro_avg", "micro"),
                **_avg_section(metrics_json, "weighted_avg", "weighted"),

                # --- VD averages. ---
                "vd_avg_input_tokens": _r(df["vd_input_tokens"].mean()),
                "vd_avg_output_tokens": _r(df["vd_output_tokens"].mean()),
                "vd_avg_total_tokens": _r(df["vd_total_tokens"].mean()),
                "vd_avg_latency_s": _r(df["vd_latency_s"].mean()),
                "vd_total_energy_kwh": _r_energy(vd_total_energy),
                "vd_total_emissions_kg": _r_energy(vd_total_emissions),
                "vd_energy_kwh_per_1k_input_tokens": _r_energy(_per_1k(vd_total_energy, vd_input_tokens)),
                "vd_energy_kwh_per_1k_output_tokens": _r_energy(_per_1k(vd_total_energy, vd_output_tokens)),
                "vd_energy_kwh_per_1k_total_tokens": _r_energy(_per_1k(vd_total_energy, vd_total_tokens)),
                "vd_total_latency_s": _r(vd_total_latency),

                # SA averages.
                "sa_avg_input_tokens": _r(df["sa_input_tokens"].mean()),
                "sa_avg_output_tokens": _r(df["sa_output_tokens"].mean()),
                "sa_avg_total_tokens": _r(df["sa_total_tokens"].mean()),
                "sa_avg_latency_s": _r(df["sa_latency_s"].mean()),
                "sa_total_energy_kwh": _r_energy(sa_total_energy),
                "sa_total_emissions_kg": _r_energy(sa_total_emissions),
                "sa_energy_kwh_per_1k_input_tokens": _r_energy(_per_1k(sa_total_energy, sa_input_tokens)),
                "sa_energy_kwh_per_1k_output_tokens": _r_energy(_per_1k(sa_total_energy, sa_output_tokens)),
                "sa_energy_kwh_per_1k_total_tokens": _r_energy(_per_1k(sa_total_energy, sa_total_tokens)),
                "sa_total_latency_s": _r(sa_total_latency),

                # Combined totals.
                "total_energy_kwh": _r_energy(combined_energy),
                "total_emissions_kg": _r_energy(combined_emissions),
                "combined_latency_s": _r(combined_latency),

                # Energy for a single execution (total VD+SA divided by the number of runs).
                "energy_kwh_per_run": _r_energy(single_execution_energy),

                # Across-run variability of energy (outlier check).
                # Computed over the per-run TOTAL energy of each run.
                "vd_energy_kwh_std": vd_e_std,
                "vd_energy_kwh_min": vd_e_min,
                "vd_energy_kwh_max": vd_e_max,
                "energy_kwh_per_run_std": per_run_e_std,
                "energy_kwh_per_run_min": tot_e_min,
                "energy_kwh_per_run_max": tot_e_max,

                # Trade-off metric: Macro F1/kWh single execution.
                "macro_f1_per_kwh": _r(
                    metrics_json["macro_avg"]["f1_mean"] / single_execution_energy
                    if single_execution_energy > 0 else 0.0
                ),

                "num_contracts": num_contracts,
                "num_runs": num_runs,
            }
            rows.append(row)

    if not rows:
        raise SystemExit("No run_*.json files found under results/<model>/<prompt>/")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinal report written to {OUT_CSV} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
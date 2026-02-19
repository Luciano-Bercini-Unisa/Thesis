import json
import csv
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_CSV = RESULTS_DIR / "final_results.csv"


def load_metrics(prompt_folder: Path):
    path = prompt_folder / "average_results.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["macro_avg"], data["per_class"]


def load_raw_csvs(prompt_folder: Path):
    csvs = list(prompt_folder.glob("*.csv"))
    dfs = [pd.read_csv(c) for c in csvs]
    return pd.concat(dfs, ignore_index=True), len(csvs)


def main():
    rows = []

    if not RESULTS_DIR.exists():
        raise SystemExit("results/ directory not found")

    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        for prompt_dir in model_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            avg_path = prompt_dir / "average_results.json"
            if not avg_path.exists():
                continue

            macro, per_class = load_metrics(prompt_dir)
            df, num_runs = load_raw_csvs(prompt_dir)

            for col in [
                "vd_energy_kwh", "vd_emissions_kg", "vd_latency_s",
                "vd_input_tokens", "vd_output_tokens", "vd_total_tokens",
                "sa_energy_kwh", "sa_emissions_kg", "sa_latency_s",
                "sa_input_tokens", "sa_output_tokens", "sa_total_tokens"
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            model_name = model_dir.name
            prompt_name = prompt_dir.name
            num_contracts = df["file_name"].nunique()

            vd_total_energy = df["vd_energy_kwh"].sum()
            sa_total_energy = df["sa_energy_kwh"].sum()

            vd_total_tokens = df["vd_total_tokens"].sum()
            sa_total_tokens = df["sa_total_tokens"].sum()

            combined_energy = vd_total_energy + sa_total_energy
            combined_tokens = vd_total_tokens + sa_total_tokens

            vd_energy_per_1k = (
                vd_total_energy / (vd_total_tokens / 1000)
                if vd_total_tokens > 0 else 0
            )

            sa_energy_per_1k = (
                sa_total_energy / (sa_total_tokens / 1000)
                if sa_total_tokens > 0 else 0
            )

            vd_total_latency = df["vd_latency_s"].sum()
            sa_total_latency = df["sa_latency_s"].sum()
            combined_latency = vd_total_latency + sa_total_latency

            vd_latency_per_1k = (
                vd_total_latency / (vd_total_tokens / 1000)
                if vd_total_tokens > 0 else 0
            )

            sa_latency_per_1k = (
                sa_total_latency / (sa_total_tokens / 1000)
                if sa_total_tokens > 0 else 0
            )

            combined_latency_per_1k = (
                combined_latency / (combined_tokens / 1000)
                if combined_tokens > 0 else 0
            )

            rows.append({
                "model": model_name,
                "prompt": prompt_name,

                "macro_precision_mean": macro["precision_mean"],
                "macro_recall_mean": macro["recall_mean"],
                "macro_f1_mean": macro["f1_mean"],
                "macro_f1_std": macro["f1_std"],

                # VD averages.
                "vd_avg_input_tokens": df["vd_input_tokens"].mean(),
                "vd_avg_output_tokens": df["vd_output_tokens"].mean(),
                "vd_avg_total_tokens": df["vd_total_tokens"].mean(),
                "vd_avg_latency_s": df["vd_latency_s"].mean(),
                "vd_total_energy_kwh": vd_total_energy,
                "vd_total_emissions_kg": df["vd_emissions_kg"].sum(),
                "vd_energy_per_1k_tokens": vd_energy_per_1k,
                # VD latency.
                "vd_total_latency_s": vd_total_latency,
                "vd_latency_per_1k_tokens": vd_latency_per_1k,
                # SA averages.
                "sa_avg_input_tokens": df["sa_input_tokens"].mean(),
                "sa_avg_output_tokens": df["sa_output_tokens"].mean(),
                "sa_avg_total_tokens": df["sa_total_tokens"].mean(),
                "sa_avg_latency_s": df["sa_latency_s"].mean(),
                "sa_total_energy_kwh": sa_total_energy,
                "sa_total_emissions_kg": df["sa_emissions_kg"].sum(),
                "sa_energy_per_1k_tokens": sa_energy_per_1k,
                # SA latency.
                "sa_total_latency_s": sa_total_latency,
                "sa_latency_per_1k_tokens": sa_latency_per_1k,
                # Combined totals.
                "total_energy_kwh": df["vd_energy_kwh"].sum() + df["sa_energy_kwh"].sum(),
                "total_emissions_kg": df["vd_emissions_kg"].sum() + df["sa_emissions_kg"].sum(),
                "combined_latency_s": combined_latency,
                "combined_latency_per_1k_tokens": combined_latency_per_1k,

                "num_contracts": num_contracts,
                "num_runs": num_runs,
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        if not rows:
            raise SystemExit("No average_results.json found under results/<model>/<prompt>/")
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Final report written to {OUT_CSV}")


if __name__ == "__main__":
    main()

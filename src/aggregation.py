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


def per_1k(energy_kwh: float, tokens: float) -> float:
    return energy_kwh / (tokens / 1000) if tokens > 0 else 0


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

            macro, _ = load_metrics(prompt_dir)
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
            # Unique contracts across runs.
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
                "vd_total_emissions_kg": vd_total_emissions,
                "vd_energy_kwh_per_1k_input_tokens": per_1k(vd_total_energy, vd_input_tokens),
                "vd_energy_kwh_per_1k_output_tokens": per_1k(vd_total_energy, vd_output_tokens),
                "vd_energy_kwh_per_1k_total_tokens": per_1k(vd_total_energy, vd_total_tokens),
                "vd_total_latency_s": vd_total_latency,
                # SA averages.
                "sa_avg_input_tokens": df["sa_input_tokens"].mean(),
                "sa_avg_output_tokens": df["sa_output_tokens"].mean(),
                "sa_avg_total_tokens": df["sa_total_tokens"].mean(),
                "sa_avg_latency_s": df["sa_latency_s"].mean(),
                "sa_total_energy_kwh": sa_total_energy,
                "sa_total_emissions_kg": sa_total_emissions,
                "sa_energy_kwh_per_1k_input_tokens": per_1k(sa_total_energy, sa_input_tokens),
                "sa_energy_kwh_per_1k_output_tokens": per_1k(sa_total_energy, sa_output_tokens),
                "sa_energy_kwh_per_1k_total_tokens": per_1k(sa_total_energy, sa_total_tokens),
                "sa_total_latency_s": sa_total_latency,
                # Combined totals.
                "total_energy_kwh": combined_energy,
                "total_emissions_kg": combined_emissions,
                "combined_latency_s": combined_latency,

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

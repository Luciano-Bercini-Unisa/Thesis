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

            df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
            df["emissions_kg"] = pd.to_numeric(df["emissions_kg"], errors="coerce")
            df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce")

            model_name = model_dir.name
            prompt_name = prompt_dir.name
            num_contracts = df["file_name"].nunique()

            rows.append({
                "model": model_name,
                "prompt": prompt_name,
                "macro_precision_mean": macro["precision_mean"],
                "macro_recall_mean": macro["recall_mean"],
                "macro_f1_mean": macro["f1_mean"],
                "macro_f1_std": macro["f1_std"],
                "avg_input_tokens": df["input_tokens"].mean(),
                "avg_output_tokens": df["output_tokens"].mean(),
                "avg_total_tokens": df["total_tokens"].mean(),
                "avg_latency_s": df["latency_s"].mean(),
                "total_energy_kwh": df["energy_kwh"].sum(),
                "total_emissions_kg": df["emissions_kg"].sum(),
                "energy_per_contract_kwh": df["energy_kwh"].sum() / num_contracts if num_contracts else 0,
                "emissions_per_contract_kg": df["emissions_kg"].sum() / num_contracts if num_contracts else 0,
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

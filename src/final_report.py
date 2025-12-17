# The experimental pipeline is divided into three distinct phases: execution, evaluation, and aggregation.

# 1. In the execution phase, each smart contract from the SmartBugs-Curated dataset is processed individually
# using an open-source large language model. For each contract, the system records inference latency,
# token usage, and energy consumption using CodeCarbon, and produces a structured prediction map.

# 2. In the evaluation phase, raw predictions are compared against the ground-truth vulnerability annotations
# to compute precision, recall, specificity, and F1-score at both per-class and macro-averaged levels.
# To mitigate model instability, each experiment is repeated five times and results are averaged.

# 3. Finally, in the aggregation phase, quality metrics are combined with energy and token statistics to
# generate a consolidated report at the model–prompt level.
# This allows direct comparison between prompt-engineering strategies in terms of
# effectiveness, efficiency, and sustainability.

import json
import csv
import glob
import os
import pandas as pd

RESULTS_DIR = "./results"
OUT_CSV = "./final_results.csv"

def load_metrics(prompt_folder):
    path = os.path.join(RESULTS_DIR, prompt_folder, "average_results.json")
    with open(path, "r") as f:
        data = json.load(f)
    return data["macro_avg"], data["per_class"]

def load_raw_csvs(prompt_folder):
    csvs = glob.glob(os.path.join(RESULTS_DIR, prompt_folder, "*.csv"))
    dfs = [pd.read_csv(c) for c in csvs]
    return pd.concat(dfs, ignore_index=True), len(csvs)

def main():
    rows = []

    for prompt in os.listdir(RESULTS_DIR):
        folder = os.path.join(RESULTS_DIR, prompt)
        if not os.path.isdir(folder):
            continue
        if not os.path.exists(os.path.join(folder, "average_results.json")):
            continue

        macro, per_class = load_metrics(prompt)
        df, num_runs = load_raw_csvs(prompt)

        df["energy_kwh_total"] = pd.to_numeric(df["energy_kwh_total"], errors="coerce")
        df["emissions_kg_total"] = pd.to_numeric(df["emissions_kg_total"], errors="coerce")
        df["latency_s"] = pd.to_numeric(df["latency_s"], errors="coerce")

        model_name = df["model_name"].iloc[0]
        num_contracts = df.shape[0] // num_runs

        rows.append({
            "model": model_name,
            "prompt": prompt,
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1_score"],
            "macro_f1_std": next(
                (c.get("f1_score_std", 0) for c in per_class if "f1_score_std" in c),
                0
            ),
            "avg_input_tokens": df["input_tokens"].mean(),
            "avg_output_tokens": df["output_tokens"].mean(),
            "avg_total_tokens": df["total_tokens"].mean(),
            "avg_latency_s": df["latency_s"].mean(),
            "total_energy_kwh": df["energy_kwh_total"].sum(),
            "total_emissions_kg": df["emissions_kg_total"].sum(),
            "energy_per_contract_kwh": df["energy_kwh_total"].sum() / num_contracts,
            "emissions_per_contract_kg": df["emissions_kg_total"].sum() / num_contracts,
            "num_contracts": num_contracts,
            "num_runs": num_runs,
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Final report written to {OUT_CSV}")

if __name__ == "__main__":
    main()

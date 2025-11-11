#apri il file json presente nella cartella results/variant_1 di nome 1_VARIANT_1_output.json
import json
import statistics
import numpy as np
import json, os, csv
import glob, argparse
from collections import defaultdict

def avg_metrics_rp(prompt_variant) -> dict:
    lower_case = prompt_variant.lower()
    # 1. Carica tutti i 5 file JSON (assumiamo siano nella cartella corrente e chiamati results_1.json, ..., results_5.json)
    file_paths = sorted(glob.glob(f"../results/{lower_case}/*_output_metrics_rp.json"))

    # Strutture dati
    class_metrics = defaultdict(lambda: defaultdict(float))          # somma dei valori
    class_f1_scores = defaultdict(list)                              # lista di f1 per std
    macro_avg = defaultdict(float)
    count = len(file_paths)

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Somma per classi + accumulo F1 per std
        for entry in data["per_class"]:
            class_name = entry["class"]
            for key, value in entry.items():
                if key != "class":
                    class_metrics[class_name][key] += value
                    if key == "f1_score":
                        class_f1_scores[class_name].append(value)

        # Macro/Micro avg
        for key, value in data["macro_avg"].items():
            macro_avg[key] += value

    # Calcolo media e std
    avg_per_class = []
    for class_name, metrics in class_metrics.items():
        avg_entry = {"class": class_name}
        for key, total in metrics.items():
            avg_entry[key] = round(total / count, 4)
        # Calcolo deviazione standard solo per f1_score
        if class_name in class_f1_scores and len(class_f1_scores[class_name]) > 1:
            std = statistics.stdev(class_f1_scores[class_name])
            avg_entry["f1_score_std"] = round(std, 4)
        avg_per_class.append(avg_entry)

    # Media macro e micro
    avg_macro = {key: round(val / count, 4) for key, val in macro_avg.items()}

    # Risultato finale
    final_result = {
        "per_class": sorted(avg_per_class, key=lambda x: x["class"]),
        "macro_avg": avg_macro,
    }

    # 6. Scrivi su file
    with open(f"../results/{lower_case}/average_results_rp.json", "w") as f:
        json.dump(final_result, f, indent=2)

def avg_metrics_scikit(prompt_variant) -> dict:
    lower_case = prompt_variant.lower()
    # 1. Carica tutti i file JSON
    file_paths = sorted(glob.glob(f"../results/{lower_case}/*_output_scikit.json"))

    # Strutture dati
    class_metrics = defaultdict(lambda: defaultdict(list))  # accumula valori per std
    macro_metrics = defaultdict(list)
    micro_metrics = defaultdict(list)
    weighted_metrics = defaultdict(list)
    count = len(file_paths)

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Per ogni classe top-level (tranne macro/micro/weighted)
        for class_name, metrics in data.items():
            if class_name in ["macro", "micro", "weighted"]:
                continue
            for key, value in metrics.items():
                class_metrics[class_name][key].append(value)

        # Macro/Micro/Weighted
        for key, value in data.get("macro", {}).items():
            macro_metrics[key].append(value)
        for key, value in data.get("micro", {}).items():
            micro_metrics[key].append(value)
        for key, value in data.get("weighted", {}).items():
            weighted_metrics[key].append(value)

    # Calcola media e std per classe
    avg_per_class = []
    for class_name, metrics_dict in class_metrics.items():
        avg_entry = {"class": class_name}
        for key, values in metrics_dict.items():
            avg_entry[key] = round(sum(values) / len(values), 2)
            if key == "f1-score" and len(values) > 1:
                std = statistics.stdev(values)
                avg_entry["f1_score_std"] = round(std, 2)
        avg_per_class.append(avg_entry)

    # Calcolo media macro/micro/weighted
    avg_macro = {key: round(sum(values) / len(values), 2) for key, values in macro_metrics.items()}
    avg_micro = {key: round(sum(values) / len(values), 2) for key, values in micro_metrics.items()}
    avg_weighted = {key: round(sum(values) / len(values), 2) for key, values in weighted_metrics.items()}

    # Risultato finale
    final_result = {
        "per_class": sorted(avg_per_class, key=lambda x: x["class"]),
        "macro": avg_macro,
        "micro": avg_micro,
        "weighted": avg_weighted,
    }

    # Salvataggio su file
    with open(f"../results/{lower_case}/average_results_scikit.json", "w") as f:
        json.dump(final_result, f, indent=2)

    return final_result

def avg_metrics(prompt_variant) -> dict:
    lower_case = prompt_variant.lower()
    # 1. Carica tutti i 5 file JSON (assumiamo siano nella cartella corrente e chiamati results_1.json, ..., results_5.json)
    file_paths = sorted(glob.glob(f"../results/{lower_case}/*_output_metrics.json"))

    # Strutture dati
    class_metrics = defaultdict(lambda: defaultdict(float))          # somma dei valori
    class_f1_scores = defaultdict(list)                              # lista di f1 per std
    macro_avg = defaultdict(float)
    micro_avg = defaultdict(float)
    count = len(file_paths)

    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Somma per classi + accumulo F1 per std
        for entry in data["per_class"]:
            class_name = entry["class"]
            for key, value in entry.items():
                if key != "class":
                    class_metrics[class_name][key] += value
                    if key == "f1_score":
                        class_f1_scores[class_name].append(value)

        # Macro/Micro avg
        for key, value in data["macro_avg"].items():
            macro_avg[key] += value
        for key, value in data["micro_avg"].items():
            micro_avg[key] += value

    # Calcolo media e std
    avg_per_class = []
    for class_name, metrics in class_metrics.items():
        avg_entry = {"class": class_name}
        for key, total in metrics.items():
            avg_entry[key] = round(total / count, 4)
        # Calcolo deviazione standard solo per f1_score
        if class_name in class_f1_scores and len(class_f1_scores[class_name]) > 1:
            std = statistics.stdev(class_f1_scores[class_name])
            avg_entry["f1_score_std"] = round(std, 4)
        avg_per_class.append(avg_entry)

    # Media macro e micro
    avg_macro = {key: round(val / count, 4) for key, val in macro_avg.items()}
    avg_micro = {key: round(val / count, 4) for key, val in micro_avg.items()}

    # Risultato finale
    final_result = {
        "per_class": sorted(avg_per_class, key=lambda x: x["class"]),
        "macro_avg": avg_macro,
        "micro_avg": avg_micro
    }

    # 6. Scrivi su file
    with open(f"../results/{lower_case}/average_results.json", "w") as f:
        json.dump(final_result, f, indent=2)

def csv_aggregate():
    # Percorso dei file JSON
    json_files = sorted(glob.glob(f"./results/*/average_results.json"))  # Es: prompt1.json, prompt2.json, prompt3.json

    # File CSV di output
    output_csv = "prompts_metrics.csv"

    # Campi da includere
    fields = ["prompt", "class", "precision", "recall", "f1_score", "f1_score_std", "specificity", "tp", "fp", "tn", "fn"]

    rows = []

    for file_path in json_files:
        prompt_name = file_path.split("/")[2]  # estrae 'variant_1' da './results/variant_1/avarage_results.json'

        with open(file_path, "r") as f:
            data = json.load(f)

        # Per-class metrics
        for cls in data.get("per_class", []):
            row = {
                "prompt": prompt_name,
                "class": cls["class"],
                "precision": cls.get("precision", ""),
                "recall": cls.get("recall", ""),
                "f1_score": cls.get("f1_score", ""),
                "f1_score_std": cls.get("f1_score_std", ""),
                "specificity": cls.get("specificity", ""),
                "tp": cls.get("tp", ""),
                "fp": cls.get("fp", ""),
                "tn": cls.get("tn", ""),
                "fn": cls.get("fn", "")
            }
            rows.append(row)

        # Macro-average
        macro = data.get("macro_avg", {})
        row_macro = {
            "prompt": prompt_name,
            "class": "macro_avg",
            "precision": macro.get("precision", ""),
            "recall": macro.get("recall", ""),
            "f1_score": macro.get("f1_score", ""),
            "f1_score_std": "",  # non disponibile
            "specificity": macro.get("specificity", ""),
            "tp": "", "fp": "", "tn": "", "fn": ""
        }
        rows.append(row_macro)

        # Micro-average
        micro = data.get("micro_avg", {})
        row_micro = {
            "prompt": prompt_name,
            "class": "micro_avg",
            "precision": micro.get("precision", ""),
            "recall": micro.get("recall", ""),
            "f1_score": micro.get("f1_score", ""),
            "f1_score_std": "",  # non disponibile
            "specificity": micro.get("specificity", ""),
            "tp": "", "fp": "", "tn": "", "fn": ""
        }
        rows.append(row_micro)

    # Scrivi il CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ CSV salvato come: {output_csv}")


# avg_metrics("VARIANT_1")
# avg_metrics("VARIANT_2")
# avg_metrics("VARIANT_3")
# avg_metrics("ORIGINAL")
# avg_metrics("FEWSHOTS")
# csv_aggregate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metrics.")
    parser.add_argument('--folderpath', type=str, help='Folder to process', required=True)
    args = parser.parse_args()
    # avg_metrics(args.filename)
    avg_metrics_rp(args.filename)
    avg_metrics_scikit(args.filename)

# avg_metrics("original/uncommented_gpt-4")
# avg_metrics_scikit("original/unlabelled_gpt-4")

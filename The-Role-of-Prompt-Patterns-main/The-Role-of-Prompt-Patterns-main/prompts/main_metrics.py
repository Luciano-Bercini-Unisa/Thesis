import json, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Lista ordinata delle vulnerabilità da usare come intestazioni (righe e colonne)
# VULNERABILITIES = [
#     "access_control",
#     "arithmetic",
#     "bad_randomness",
#     "denial_of_service",
#     "front_running",
#     "reentrancy",
#     "short_addresses",
#     "time_manipulation",
#     "unchecked_low_level_calls"
# ]

VULNERABILITIES = [
    "access_control",
    "arithmetic",
    "bad_randomness",
    "denial_of_service",
    "front_running",
    "reentrancy",
    "short_addresses",
    "time_manipulation",
    "unchecked_low_level_calls"
]

# Mappiamo tutto in lowercase con underscore per coerenza
def normalize(vuln):
    return vuln.strip().lower().replace(" ", "_")

def parse_findings(findings_raw):
    findings = {}
    for line in findings_raw.strip().split("\n"):
        if ": " in line:
            key, val = line.split(": ", 1)
            try:
                findings[normalize(key.strip())] = int(val.strip())
            except ValueError:
                print(f"Warning: could not parse count for line: {line}")
    return findings


# Creazione della matrice
def build_confusion_matrix(data):
    size = len(VULNERABILITIES)
    matrix = np.zeros((size, size), dtype=int)

    index_map = {vuln: i for i, vuln in enumerate(VULNERABILITIES)}

    for item in data:
        true_label = normalize(item["category"])
        findings = parse_findings(item["findings"])

        if true_label not in index_map:
            continue  # salta se la categoria vera non è tra quelle riconosciute

        row = index_map[true_label]
        for pred_label, val in findings.items():
            if val == 1 and pred_label in index_map:
                col = index_map[pred_label]
                matrix[row][col] += 1

    return matrix

# Visualizzazione della matrice
def plot_matrix(matrix, labels, output_filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual (Ground Truth)")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"../results/"+output_filename+"_matrix.png", dpi=300)


def calcola_metriche(matrix):
    results = []
    total_tp = total_fp = total_fn = total_tn = 0

    for vuln in VULNERABILITIES:
        #estrai la riga dalla matrice in corrispondenza di vuln usando il nome della vulnerabilità
        row = matrix[VULNERABILITIES.index(vuln)]
        col_index = VULNERABILITIES.index(vuln)
        column = [int(row[col_index]) for row in matrix]
        print(sum(row))
        print(sum(column))
        # Calcola le metriche
        tp = row[VULNERABILITIES.index(vuln)]
        fp = sum(column) - tp
        fn = sum(row) - tp
        total = matrix.sum()
        tn = total - tp - fp - fn 

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp)  

        results.append({
            'class': vuln,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'precision': round(float(precision), 2),
            'recall': round(float(recall), 2),
            'specificity': round(float(specificity), 2),
            'f1_score': round(float(f1), 2)
        })
    total_tn = matrix.sum() - total_tp - total_fp - total_fn
    # Macro average
    macro_precision = sum(c['precision'] for c in results) / len(results)
    macro_recall = sum(c['recall'] for c in results) / len(results)
    macro_f1 = sum(c['f1_score'] for c in results) / len(results)
    macro_specificity = sum(c['specificity'] for c in results) / len(results)

    # Micro average
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0
    micro_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0

    return {
        'per_class': results,
        'macro_avg': {
            'precision': round(float(macro_precision), 2),
            'recall': round(float(macro_recall), 2),
            'specificity': round(float(macro_specificity), 2),
            'f1_score': round(float(macro_f1), 2)
        },
        'micro_avg': {
            'precision': round(float(micro_precision), 2),
            'recall': round(float(micro_recall), 2),
            'specificity': round(float(micro_specificity), 2),
            'f1_score': round(float(micro_f1), 2)
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metrics.")
    parser.add_argument('--filename', type=str, help='Filename to process', required=True)
    args = parser.parse_args()

    with open(f"../results/{args.filename}.json", "r") as f:
        data = json.load(f)

    matrix = build_confusion_matrix(data)
    metrics = calcola_metriche(matrix)
    with open(f"../results/{args.filename}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(metrics)
    plot_matrix(matrix, VULNERABILITIES, args.filename)

# python3 main_metrics.py --filename 1_VARIANT_1_output
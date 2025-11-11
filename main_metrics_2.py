
import json, argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

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

def compute_metrics_from_confusion(matrix, class_labels):
    # Ricostruisci y_true e y_pred dalla matrice NxN
    y_true = []
    y_pred = []
    for i in range(matrix.shape[0]):       # righe = vere
        for j in range(matrix.shape[1]):   # colonne = predette
            count = matrix[i, j]
            y_true.extend([class_labels[i]] * count)
            y_pred.extend([class_labels[j]] * count)

    # Report completo
    print("\nClassification report:\n")
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        zero_division=0,
        output_dict=True  # <---- chiave importante
    )
    # rimuove le chiavi non volute
    for key in ["accuracy", "macro avg", "weighted avg"]:
        report_dict.pop(key, None)
    for label in report_dict.keys():
        if isinstance(report_dict[label], dict) and "support" in report_dict[label]:
            report_dict[label].pop("support")
    print(report_dict)
    # Converti in JSON string (opzionale)
    report_json = json.dumps(report_dict, indent=4)

    # Oppure metriche aggregate
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # Aggiungi queste metriche al dict
    report_dict["macro"] = {
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro    = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro        = f1_score(y_true, y_pred, average='micro', zero_division=0)
    # Aggiungi queste metriche al dict
    report_dict["micro"] = {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro
    }
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted        = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    # Aggiungi queste metriche al dict
    report_dict["weighted"] = {
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }
    print(f"Macro Avg - Precision: {precision_macro:.3f}, Recall: {recall_macro:.3f}, F1: {f1_macro:.3f}")
    print(f"Micro Avg - Precision: {precision_micro:.3f}, Recall: {recall_micro:.3f}, F1: {f1_micro:.3f}")
    print(f"Weighted Avg - Precision: {precision_weighted:.3f}, Recall: {recall_weighted:.3f}, F1: {f1_weighted:.3f}")

    return report_dict

# def compute_metrics_from_confusion(matrix, class_labels):
#     n_classes = len(class_labels)
    
#     # Calcolo TP, FP, FN, TN per classe
#     TP = np.diag(matrix)
#     FP = matrix.sum(axis=0) - TP
#     FN = matrix.sum(axis=1) - TP
#     TN = matrix.sum() - (TP + FP + FN)
    
#     precision_per_class = TP / (TP + FP + 1e-10)  # evita divisione per 0
#     recall_per_class    = TP / (TP + FN + 1e-10)
#     f1_per_class        = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-10)
    
#     # Stampa per classe
#     print("Metrics per class:")
#     for i, label in enumerate(class_labels):
#         print(f"{label}: Precision={precision_per_class[i]:.3f}, Recall={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}")
    
#     # Macro average
#     precision_macro = np.mean(precision_per_class)
#     recall_macro = np.mean(recall_per_class)
#     f1_macro = np.mean(f1_per_class)
    
#     # Micro average
#     tp_total = TP.sum()
#     fp_total = FP.sum()
#     fn_total = FN.sum()
#     precision_micro = tp_total / (tp_total + fp_total + 1e-10)
#     recall_micro = tp_total / (tp_total + fn_total + 1e-10)
#     f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-10)
    
#     print(f"\nMacro Avg - Precision: {precision_macro:.3f}, Recall: {recall_macro:.3f}, F1: {f1_macro:.3f}")
#     print(f"Micro Avg - Precision: {precision_micro:.3f}, Recall: {recall_micro:.3f}, F1: {f1_micro:.3f}")
    
#     return precision_per_class, recall_per_class, f1_per_class, precision_macro, recall_macro, f1_macro, precision_micro, recall_micro, f1_micro

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metrics.")
    parser.add_argument('--filename', type=str, help='Filename to process', required=True)
    args = parser.parse_args()

    with open(f"../results/{args.filename}.json", "r") as f:
        data = json.load(f)

    matrix = build_confusion_matrix(data)
    metrics = compute_metrics_from_confusion(matrix, VULNERABILITIES)
    with open(f"../results/{args.filename}_scikit.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(metrics)

# python3 main_metrics2.py --filename 1_VARIANT_1_output

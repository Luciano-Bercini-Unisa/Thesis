# Checks which contracts were perfectly classified by matching
# the model's semantic-analysis predictions against the SmartBugs ground truth.
# No output other than printing to console (debug utility).
import argparse
import glob
import json
from ground_truth_extraction import extract_ground_truth


def load_run_json(path):
    with open(path, "r") as file_r:
        return json.load(file_r)


def analyze_perfect_detection(run_json, gt):
    """
        Compare each contract's predicted vulnerability map to the ground truth.
        run_json: list of items produced by measure_multi_test.py,
                  each containing 'file_name' and 'prediction_map'.
        ground_truth: dict mapping file_name → dict of vulnerability flags.
        Returns:
            A dict where keys are file names and values are True
            if the model produced a perfect match for that contract.
        """
    result = {}
    # Iterate over all predictions from one run.
    for item in run_json:
        file_name = item["file_name"]
        pred_map = item["prediction_map"]
        # Skip files that don't exist in the ground truth.
        if file_name not in gt:
            continue
        gt_map = gt[file_name]
        # Perfect match?
        if pred_map == gt_map:
            result[file_name] = True
    return result


if __name__ == "__main__":
    # Parse argument: which folder (inside results/) to evaluate.
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True,
                        help="Folder inside results/, e.g. ORIGINAL or VARIANT_1")
    args = parser.parse_args()
    # Load SmartBugs ground truth once.
    ground_truth = extract_ground_truth()
    # Collect all *_output.json result files in the chosen folder.
    run_files = sorted(glob.glob(f"./results/{args.folder}/*_output.json"))
    # Track perfect detections across all runs.
    # Initialize all contracts as 'not perfectly detected'.
    overall = {fn: False for fn in ground_truth}
    # Process each run file.
    for rf in run_files:
        run_data = load_run_json(rf)
        perfect = analyze_perfect_detection(run_data, ground_truth)
        # Mark contracts detected perfectly in *any* run.
        for file_name in perfect:
            overall[file_name] = True

    # Print summary.
    perfect_count = sum(1 for v in overall.values() if v)
    print(f"Perfectly detected contracts: {perfect_count}")
    for fn, ok in overall.items():
        if ok:
            print("  -", fn)
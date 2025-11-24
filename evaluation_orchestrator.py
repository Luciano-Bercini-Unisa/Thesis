"""
Orchestrates the full evaluation pipeline:
1. Extract ground truth from SmartBugs-Curated.
2. Compute perfect detections.
3. Compute aggregated metrics (precision/recall/F1/etc.)
"""

import subprocess
import argparse

def run(cmd):
    print(f"\n=== Running: {cmd} ===")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True,
                        help="Name of results/<folder> to analyze")
    args = parser.parse_args()

    run(f"python debug_perfect_detection.py --folder {args.folder}")
    run(f"python metrics.py --folder {args.folder}")

    print("\n=== Pipeline completed successfully ===")
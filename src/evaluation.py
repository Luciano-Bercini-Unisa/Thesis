"""
Orchestrates the full evaluation pipeline:
1. Compute perfect detections (debug utility) from SmartBugs-curated.
2. Compute aggregated metrics (precision / recall / F1 / confusion matrices).
"""

import os
import subprocess
import argparse
import sys

ROOT = os.path.dirname(__file__)

def run_py(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", f"src.{module}", *args]
    print("\n=== Running ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Model folder inside results/ (e.g. microsoft__Phi-3.5-mini-instruct)"
    )
    parser.add_argument("--prompt", required=True, help="Prompt key (e.g. ZS, ZS_COT)")
    args = parser.parse_args()

    run_py("metrics", ["--model", args.model, "--prompt", args.prompt])

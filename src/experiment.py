# experiment.py
# Single entry point: execution (N runs) -> evaluation -> aggregation

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

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_py(script: str, args: list[str]) -> None:
    script_path = ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    cmd = [sys.executable, str(script_path), *args]
    print("\n=== Running ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run full experiment pipeline: execution -> evaluation -> aggregation"
    )
    ap.add_argument("--dataset", required=True, help="Path to SmartBugs-Curated root")
    ap.add_argument("--model", required=True, help="HF model name (e.g. microsoft/Phi-3.5-mini-instruct)")
    ap.add_argument("--prompt", required=True, help="Prompt key (e.g. ZS, ZS_COT, FS)")
    ap.add_argument("--persona", action="store_true", help="Enable VD persona (system prompt)")
    ap.add_argument("--runs", type=int, default=5, help="Number of repeated runs")
    ap.add_argument("--sa_prompt", default="SA", help="Semantic analysis prompt key (e.g. SA, STRUCTURED_SA)")

    ap.add_argument("--strip_comments", action="store_true", help="Strip Solidity comments (default: keep)")
    ap.add_argument("--resume", action="store_true", help="Resume from last run JSON (best-effort)")
    ap.add_argument("--skip_evaluation", action="store_true")
    ap.add_argument("--skip_aggregation", action="store_true")
    args = ap.parse_args()

    safe_model = args.model.replace("/", "__")
    if args.persona:
        effective_prompt = f"{args.prompt}_PERSONA"
    else:
        effective_prompt = args.prompt
    # 1. Execution (repeat N times).
    for i in range(args.runs):
        resume_json = ""
        if args.resume:
            base_dir = ROOT / "results" / safe_model / effective_prompt
            # Resume from the most recent run_*.json, if any.
            runs = sorted(base_dir.glob("run_*.json"))
            if runs:
                resume_json = str(runs[-1])

        exec_args = [
            "--model", args.model,
            "--dataset", args.dataset,
            "--prompt", args.prompt,
            "--sa_prompt", args.sa_prompt,
        ]
        if args.persona:
            exec_args.append("--persona")
        if args.strip_comments:
            exec_args.append("--strip_comments")
        if resume_json:
            exec_args.extend(["--resume_json", resume_json])

        run_py("execution.py", exec_args)
    # 2. Evaluation.
    if not args.skip_evaluation:
        run_py("evaluation.py", ["--model", safe_model, "--prompt", effective_prompt])

    # 3. Aggregation (global over all models/prompts)
    if not args.skip_aggregation:
        run_py("aggregation.py", [])

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

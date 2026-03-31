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
# generate a consolidated report at the model-prompt level.
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
    ap.add_argument("--model", required=True, help="HF model name (e.g. microsoft/Phi-3.5-mini-instruct)")
    ap.add_argument("--dataset", required=True, help="Path to SmartBugs-Curated root")
    ap.add_argument("--prompt", required=True, help="Prompt key (e.g. ZS, ZS_COT, FS)")
    ap.add_argument("--role", action="store_true", help="Enable VD role (system prompt)")
    ap.add_argument("--runs", type=int, default=5, help="Number of repeated runs")
    ap.add_argument("--sa_prompt", default="SA", help="Semantic analysis prompt key, used only when parser_mode='sa'")
    ap.add_argument("--no_strip_comments", action="store_false", dest="strip_comments")
    ap.set_defaults(strip_comments=True)
    ap.add_argument(
        "--parser_mode",
        default="sa",
        choices=["deterministic", "sa"],
        help="How to convert VD output into the final binary prediction map."
    )
    ap.add_argument("--skip_evaluation", action="store_true")
    ap.add_argument("--skip_aggregation", action="store_true")
    args = ap.parse_args()

    safe_model = args.model.replace("/", "__")
    if args.role:
        effective_prompt = f"{args.prompt}_ROLE"
    else:
        effective_prompt = args.prompt
    effective_prompt = f"{effective_prompt}_{args.parser_mode.upper()}"
    # 1. Execution (repeat N times).
    for i in range(args.runs):
        exec_args = [
            "--model", args.model,
            "--dataset", args.dataset,
            "--prompt", args.prompt,
            "--sa_prompt", args.sa_prompt,
            "--parser_mode", args.parser_mode,
        ]
        if args.role:
            exec_args.append("--role")
        if not args.strip_comments:
            exec_args.append("--no_strip_comments")

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

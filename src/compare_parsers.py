# compare_parsers.py
# Compare deterministic parsing vs Semantic Analysis parsing on the SAME saved VD outputs.
# This script reads an existing run_XXX.json produced by execution.py, reuses each detection_output,
# applies both post-processing methods, and saves a comparison JSON.

import argparse
import json
import csv
from pathlib import Path

from .execution import (
    load_model,
    run_semantic_analysis,
    parse_vd_output,
    parse_sa_output,
    get_prompt,
    SA_PROMPT_TEMPLATES_MAP,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_json",
        required=True,
        help="Path to an existing run_XXX.json produced by execution.py"
    )
    ap.add_argument(
        "--model",
        required=True,
        help="HF model name used to run SA on the saved VD outputs"
    )
    ap.add_argument(
        "--sa_prompt",
        default="SA",
        choices=sorted(SA_PROMPT_TEMPLATES_MAP.keys()),
        help="Semantic analysis prompt key"
    )
    ap.add_argument(
        "--output_json",
        default=None,
        help="Optional output path. Default: <input>_parser_comparison.json"
    )
    ap.add_argument(
        "--output_csv",
        default=None,
        help="Optional output path. Default: <input>_parser_comparison.csv"
    )
    return ap.parse_args()


def positive_labels(prediction_map: dict) -> str:
    return "; ".join(label for label, value in prediction_map.items() if value == 1)


def excerpt(text: str, max_len: int = 300) -> str:
    text = " ".join(text.split())
    return text[:max_len] + ("..." if len(text) > max_len else "")


def main():
    args = parse_args()

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input JSON: {input_path}")

    if args.output_json:
        output_path = Path(args.output_json)
    else:
        output_path = input_path.with_name(input_path.stem + "_parser_comparison.json")

    if args.output_csv:
        csv_output_path = Path(args.output_csv)
    else:
        csv_output_path = input_path.with_name(input_path.stem + "_parser_comparison.csv")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Loaded {len(data)} entries from {input_path}")

    print(f"Loading model: {args.model}")
    tokenizer, model = load_model(args.model)
    sa_template = SA_PROMPT_TEMPLATES_MAP[args.sa_prompt]

    results = []
    csv_rows = []
    for i, item in enumerate(data, start=1):
        vd_reply = item["detection_output"]

        deterministic_prediction_map, deterministic_parsed_labels = parse_vd_output(vd_reply)

        sa_prompt = get_prompt(sa_template, vd_reply)
        sa_in_t, sa_out_t, sa_secs, sa_reply = run_semantic_analysis(
            tokenizer,
            model,
            sa_prompt
        )
        sa_prediction_map = parse_sa_output(sa_reply)

        results.append({
            "model_name": item.get("model_name"),
            "prompt_key": item.get("prompt_key"),
            "category": item.get("category"),
            "file_name": item.get("file_name"),
            "detection_output": vd_reply,

            "deterministic_prediction_map": deterministic_prediction_map,
            "deterministic_parsed_labels": sorted(deterministic_parsed_labels),
            "deterministic_parsed_count": len(deterministic_parsed_labels),

            "sa_prompt_key": args.sa_prompt,
            "sa_prompt": sa_prompt,
            "sa_output": sa_reply,
            "sa_prediction_map": sa_prediction_map,
            "sa_input_tokens": sa_in_t,
            "sa_output_tokens": sa_out_t,
            "sa_total_tokens": sa_in_t + sa_out_t,
            "sa_latency_s": sa_secs,
        })

        deterministic_positive = positive_labels(deterministic_prediction_map)
        sa_positive = positive_labels(sa_prediction_map)

        disagreement_labels = [
            vuln
            for vuln in deterministic_prediction_map.keys()
            if deterministic_prediction_map.get(vuln, 0) != sa_prediction_map.get(vuln, 0)
        ]

        csv_rows.append({
            "model_name": item.get("model_name"),
            "prompt_key": item.get("prompt_key"),
            "category": item.get("category"),
            "file_name": item.get("file_name"),

            "deterministic_parsed_count": len(deterministic_parsed_labels),
            "deterministic_parsed_labels": "; ".join(sorted(deterministic_parsed_labels)),

            "deterministic_positive_labels": deterministic_positive,
            "sa_positive_labels": sa_positive,

            "num_disagreements": len(disagreement_labels),
            "disagreement_labels": "; ".join(disagreement_labels),

            "sa_input_tokens": sa_in_t,
            "sa_output_tokens": sa_out_t,
            "sa_total_tokens": sa_in_t + sa_out_t,
            "sa_latency_s": sa_secs,

            "detection_output_excerpt": excerpt(vd_reply),
        })

        print(f"[{i}/{len(data)}] Compared parsers for {item.get('file_name')}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved comparison JSON: {output_path}")

    with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved comparison CSV: {csv_output_path}")


if __name__ == "__main__":
    main()
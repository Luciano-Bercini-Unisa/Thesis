# Given a model, a dataset, and a prompt key (prompt pattern/template to use),
# this script automatically runs a complete LLM-based vulnerability-detection experiment
# and measures its computational footprint.
# Each template contains text such as “Detect and describe vulnerabilities in the following Solidity contract: {input}”.
# It also runs Semantic Analysis of the previous detection output, but for now it's not measured.
# The output are 2:
# 1. CSV with various stats, including sustainability stats.
# 2. JSON with various stats, most importantly the prediction map (for quality evaluation).

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import glob
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time, csv, os, pathlib
import json
import re
import argparse

from difflib import get_close_matches
from vulnerabilities_constants import CATEGORIES, KEYS_TO_CATEGORIES
from prompts import (
    # Include the vulnerability-detection templates.
    ORIGINAL_PROMPT_VD, PROMPT_STRUCTURED_VD,
    PROMPT_STRUCTURED_VD_VARIANT_1, PROMPT_STRUCTURED_VD_VARIANT_2, PROMPT_STRUCTURED_VD_VARIANT_3,
    PROMPT_VD_FEW_SHOT, PROMPT_VD_FEW_SHOT_WITH_EXPLANATION,
    # Include also the Semantic Analysis templates.
    ORIGINAL_PROMPT_SA, PROMPT_STRUCTURED_SA
)

# ---------- config ----------
# If GPU CUDA available then use bfloat16 (b stands for Brain in Google Brain), else float32.
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "2048")) # Just a protection against endless/degenerate loops.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

PROMPT_TEMPLATES = {
    "ORIGINAL": ORIGINAL_PROMPT_VD,
    "STRUCTURED": PROMPT_STRUCTURED_VD,
    "VARIANT_1": PROMPT_STRUCTURED_VD_VARIANT_1,
    "VARIANT_2": PROMPT_STRUCTURED_VD_VARIANT_2,
    "VARIANT_3": PROMPT_STRUCTURED_VD_VARIANT_3,
    "FEW_SHOT": PROMPT_VD_FEW_SHOT,
    "FEW_SHOT_WITH_EXPLANATION": PROMPT_VD_FEW_SHOT_WITH_EXPLANATION,
}

SA_PROMPT_TEMPLATES_MAP = {
    "SA": ORIGINAL_PROMPT_SA,
    "STRUCTURED_SA": PROMPT_STRUCTURED_SA,
}


def strip_solidity_comments(src: str) -> str:
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)  # Removes block comments.
    src = re.sub(r"//.*", "", src)  # Remove line comments.
    # Converts multiple consecutive empty/whitespace-only lines into a single blank line (double newline).
    src = re.sub(r"\n\s*\n+", "\n\n", src)
    return src.strip() # Removes leading and trailing whitespace.


def load_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE, device_map="auto").eval()
    return tok, mdl


def get_prompt(prompt_template: str, code: str) -> str:
    # The templates use "{input}".
    return prompt_template.replace("{input}", code)


def run_one_inference(tokenizer, mod, system_prompt: str | None, user_prompt: str):
    msgs = []
    # System prompt = the instruction that defines the model’s role or behavior.
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    # Get the token ids and the attention mask.
    input_tensors = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors="pt", padding=True).to(mod.device)
    # Generation settings. Sampling only if temperature > 0 (not all models support temperature;
    # doesn't matter but, it might show warnings).
    output_generation_settings = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=(TEMPERATURE > 0),
                                      temperature=TEMPERATURE, top_p=TOP_P)
    # Run generation without grad, measure latency seconds.
    t0 = time.time()
    # Disable gradient tracking (as we run only inference).
    with torch.no_grad():
        out = mod.generate(**input_tensors, **output_generation_settings)
    dt = time.time() - t0
    in_len = input_tensors["input_ids"].shape[-1]
    output_ids = out[0][in_len:]
    output_text = tokenizer.decode(output_ids)
    return in_len, output_ids.numel(), dt, output_text


def run_semantic_analysis(tokenizer, mod, sa_user_prompt: str):
    """
    Runs the semantic analyzer prompt on the free-form detection_text
    and returns (input_tokens, output_tokens, latency_s, sa_reply_text).
    """
    in_len, out_len, secs, sa_text = run_one_inference(
        tokenizer,
        mod,
        system_prompt=None, # ORIGINAL_PROMPT_SA already includes ROLE_SA etc.
        user_prompt=sa_user_prompt
    )
    return in_len, out_len, secs, sa_text.strip()


def last_emissions_row(csv_path):
    """
    Return (energy_kwh, emissions_kg) from the last row of a CodeCarbon emissions.csv file.
    Returns (None, None) if the file is missing, empty, or malformed.
    """
    # File not found or unreadable.
    try:
        with open(csv_path, encoding="utf-8") as csv_file:
            lines = [ln.strip() for ln in csv_file if ln.strip()]
    except FileNotFoundError:
        return None, None
    except OSError:
        return None, None

    # Not enough data (header only or empty).
    if len(lines) <= 1:
        return None, None

    header = lines[0].split(",")
    last_line = lines[-1].split(",")

    # Missing required fields → treat as malformed.
    try:
        e_idx = header.index("emissions")
        k_idx = header.index("energy_consumed")
    except ValueError:
        return None, None

    # Malformed numeric values → treat as missing.
    try:
        emissions_kg = float(last_line[e_idx])
        energy_kwh = float(last_line[k_idx])
    except (ValueError, IndexError):
        return None, None

    return energy_kwh, emissions_kg


# ------------ SA PARSING. ------------
# Map lowercase canonical names → proper category names.
CANONICAL = {name.lower(): name for name in CATEGORIES}

# Add synonym normalization
SYNONYMS = {
    "short address attack": "Short Addresses",
    "arithmetic issues": "Arithmetic",
    "integer overflow": "Arithmetic",
    "unchecked return values": "Unchecked Low Level Calls",
    "dos": "Denial Of Service",
    "denial of service": "Denial Of Service",
}

def normalize_name(s):
    s = s.lower().strip()
    # Strip markdown.
    s = re.sub(r"[*_`]+", "", s)
    # Strip numbering (e.g., "1. Foo").
    s = re.sub(r"^\d+\.\s*", "", s)
    # Remove parentheses content.
    s = re.sub(r"\([^)]*\)", "", s).strip()
    # Direct synonyms.
    if s in SYNONYMS:
        return SYNONYMS[s]
    # Exact canonical match.
    if s in CANONICAL:
        return CANONICAL[s]

    # fuzzy match to canonical names
    match = get_close_matches(s, CANONICAL.keys(), n=1, cutoff=0.6)
    if match:
        return CANONICAL[match[0]]

    return None

# Extracts every colon-separated pair anywhere in the text
# Strips Markdown bold "**Short Address Attack**"
# Handles lists "1. Reentrancy: 1"
# Converts synonyms "Short Address Attack" -> Short Addresses"
# Fuzzy matches "Arithmetic Issues (Integer Overflow)"
# Ignores junk tokens like <|im_end|>
# Always returns a clean dict over KEYS
def parse_sa_output(sa_text: str):
    prediction_map = {name: 0 for name in CATEGORIES}
    # Find ALL "left: digit" pairs anywhere in the output.
    pairs = re.findall(r"([A-Za-z0-9 ()_\-]+)\s*:\s*([01])", sa_text)
    for raw_name, val in pairs:
        val = int(val)
        norm = normalize_name(raw_name)
        if norm:
            prediction_map[norm] = val

    return prediction_map

# ------------ End. ------------

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--dataset", required=True, help="Path to smartbugs-curated root")
    ap.add_argument("--prompt", required=True, choices=sorted(PROMPT_TEMPLATES.keys()))
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--system", default="You are a vulnerability detector for a smart contract.")
    ap.add_argument("--no_strip_comments", action="store_false", default=True, dest="strip_comments")
    ap.add_argument("--resume_json", help="Optional JSON to resume and skip processed files")
    ap.add_argument("--sa_prompt", default="SA", choices=sorted(SA_PROMPT_TEMPLATES_MAP.keys()))
    args = ap.parse_args()
    # --------- AUTO-ASSIGN OUTPUT CSV IF NOT PROVIDED ---------
    if args.out_csv is None:
        run_dir = f"./results/{args.prompt}"
        os.makedirs(run_dir, exist_ok=True)
        run_id = len(glob.glob(f"{run_dir}/*_output.json")) + 1
        args.out_csv = f"{run_dir}/{run_id}.csv"
    # -----------------------------------------------------------
    print(f"\nLoading model: {args.model}")
    tokenizer, model = load_model(args.model)
    # Warm-up to stabilize clock/caching.
    print("Running warm-up inference...")
    _ = run_one_inference(tokenizer, model, None, "Warm up.")
    # Dedicated directory for CodeCarbon tracker for this model.
    safe_m = args.model.replace("/", "__")
    out_dir = f".codecarbon_{safe_m}"
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    # CSV.
    file_output = open(args.out_csv, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_output)
    # HEADERS.
    writer.writerow([
        "model_name", "prompt_key", "full_prompt", "category", "file_name",
        "input_tokens", "output_tokens", "total_tokens",
        "latency_s", "energy_kwh_total", "emissions_kg_total",
        "energy_kwh_per_sample", "emissions_kg_per_sample",
        "reply", "sa_prompt", "sa_reply"
    ])
    processed = set()
    if args.resume_json and pathlib.Path(args.resume_json).exists():
        with open(args.resume_json, encoding="utf-8") as f:
            try:
                prev = json.load(f)
                processed = {(x.get("category"), x.get("file_name")) for x in prev}
            except json.JSONDecodeError:
                processed = set()
    tpl = PROMPT_TEMPLATES[args.prompt]
    sa_template = SA_PROMPT_TEMPLATES_MAP[args.sa_prompt]
    sys_p = args.system.strip() if args.system else None
    json_results = []
    for cat_key, cat_name in KEYS_TO_CATEGORIES.items():
        category_directory = os.path.join(args.dataset, str(cat_key))
        print(f"Analyzing files of vulnerability category: {cat_name}")
        if not os.path.isdir(category_directory):
            continue
        for file_name in os.listdir(category_directory):
            if not file_name.endswith(".sol"):
                continue
            key = (cat_name, file_name)
            if key in processed:
                continue
            with open(os.path.join(category_directory, file_name), encoding="utf-8") as f:
                code = f.read()
            if args.strip_comments:
                code = strip_solidity_comments(code)
            user_prompt = get_prompt(tpl, code)
            # ------- Per-inference Tracking -------
            tracker = EmissionsTracker(
                measure_power_secs=10,
                output_dir=out_dir,
                save_to_file=True,
                project_name=safe_m,
                experiment_id=f"{cat_key}/{file_name}",  # Helps identify rows in emissions.csv
                log_level="error"
            )
            # Right now, the CodeCarbon tracker still measures only the detection step, not the SA call.
            # That’s fine for the moment if your priority is wiring the semantics.
            # If later you want energy of the whole pipeline (detection + SA), just move tracker.stop()
            # to after run_semantic_analysis so it wraps both calls.
            tracker.start()
            in_t, out_t, secs, detection_text = run_one_inference(tokenizer, model, sys_p, user_prompt)
            emissions_kg_this = tracker.stop()  # Per-inference kg CO2e.
            # Read energy_kwh for *this* row (the last appended row).
            energy_kwh_this, emissions_kg_csv = last_emissions_row(pathlib.Path(out_dir, "../emissions.csv"))
            if emissions_kg_csv is not None:
                emissions_kg_this = emissions_kg_csv
            # -------------------------------------
            # --- semantic analysis step (second prompt) ---
            sa_user_prompt = get_prompt(sa_template, detection_text)
            sa_in_t, sa_out_t, sa_secs, sa_text = run_semantic_analysis(
                tokenizer,
                model,
                sa_user_prompt
            )
            # Parse semantic analyzer output → dict ('access_control': 0/1, ...)
            prediction_map = parse_sa_output(sa_text)
            # Build JSON result entry.
            json_results.append({
                "model_name": args.model,
                "prompt_key": args.prompt,
                "category": cat_name,
                "file_name": file_name,
                "detection_output": detection_text,
                "semantic_output": sa_text,
                "prediction_map": prediction_map,  # <--- 0/1 dictionary.
            })
            # CSV row.
            writer.writerow([
                args.model, args.prompt, user_prompt, cat_name, file_name,
                in_t, out_t, in_t + out_t,
                f"{secs:.6f}",
                f"{energy_kwh_this:.9f}" if energy_kwh_this is not None else "",
                f"{emissions_kg_this:.9f}" if emissions_kg_this is not None else "",
                "", "",  # per-sample already recorded, so leave “per_sample” cols empty or duplicate.
                detection_text, sa_user_prompt, sa_text
            ])
            file_output.flush()  # Ensure rows land even if interrupted.
            torch.cuda.empty_cache()
    file_output.close()
    # ---- Save one JSON per run ----
    run_dir = f"./results/{args.prompt}"
    os.makedirs(run_dir, exist_ok=True)
    # Count existing runs.
    existing = len(glob.glob(f"{run_dir}/*_output.json"))
    run_id = existing + 1
    json_out = f"{run_dir}/{run_id}_output.json"
    with open(json_out, "w", encoding="utf-8") as jf:
        json.dump(json_results, jf, indent=2)
    print(f"Saved JSON results -> {json_out}")
    print(f"Done -> {args.out_csv}")


if __name__ == "__main__":
    main()

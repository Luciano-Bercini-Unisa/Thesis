# Given a model, a dataset, and a prompt key (prompt pattern/template to use),
# this script automatically runs a complete LLM-based vulnerability-detection experiment
# and measures its computational footprint.
# The output is a directory for the given prompt, which contains a CSV for each run.
# Each CSV has stats about each contract's inference.

# After that, it runs Semantic Analysis of the previous detection output.
# The output is a JSON with, among other stats, the prediction map (for quality evaluation).

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch, time, csv, os, pathlib
import json
import re
import argparse
from pathlib import Path
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM
from difflib import get_close_matches
from vulnerabilities_constants import CATEGORIES, KEYS_TO_CATEGORIES
from prompts import (
    # Include the vulnerability-detection templates.
    ZS, ZS_COT, FS, ROLE_VD,
    # Include also the Semantic Analysis templates.
    SA, ROLE_SA,
)

# If GPU CUDA is available, then use bfloat16 (b stands for Brain in Google Brain), otherwise use float32.
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
VD_MAX_NEW_TOKENS = int(os.getenv("VD_MAX_NEW_TOKENS", "1024"))
SA_MAX_NEW_TOKENS = int(os.getenv("SA_MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

PROMPT_TEMPLATES = {
    "ZS": ZS,
    "ZS_COT": ZS_COT,
    "FS": FS,
}

SA_PROMPT_TEMPLATES_MAP = {
    "SA": SA,
}


def strip_solidity_comments(src: str) -> str:
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)  # Removes block comments.
    src = re.sub(r"//.*", "", src)  # Remove line comments.
    # Converts multiple consecutive empty/whitespace-only lines into a single blank line (double newline).
    src = re.sub(r"\n\s*\n+", "\n\n", src)
    return src.strip()  # Removes leading and trailing whitespace.


def load_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = (AutoModelForCausalLM.from_pretrained(
        model_name, dtype=DTYPE, device_map="auto"
    ).eval())
    return tok, mdl


def get_prompt(prompt_template: str, code: str) -> str:
    # The templates use "{input}".
    return prompt_template.replace("{input}", code)


def run_one_inference(tokenizer, mod, system_prompt, user_prompt, max_new_tokens, temperature, top_p):
    if supports_chat(tokenizer):
        return run_chat_inference(tokenizer, mod, system_prompt, user_prompt, max_new_tokens, temperature, top_p)
    else:
        return run_sanity_inference(tokenizer, mod, user_prompt)


def run_chat_inference(tokenizer, mod, system_prompt: str | None, user_prompt: str, max_new_tokens, temperature, top_p):
    msgs = []
    # System prompt = the instruction that defines the model’s role or behavior. Could be missing.
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    # Get the token ids and the attention mask.
    input_tensors = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        truncation=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(mod.device)
    # Generation settings. Sampling only if temperature > 0 (not all models support temperature).
    gen_kwargs = dict(
        **input_tensors,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        use_cache=True
    )
    # Run generation without the grad, measure latency seconds.
    t0 = time.time()
    with torch.inference_mode():
        # Offloaded to save memory (as it was going into OOM).
        # Check: https://huggingface.co/docs/transformers/en/kv_cache
        if mod.device.type == "cuda":
            gen_kwargs["cache_implementation"] = "offloaded"
        out = mod.generate(**gen_kwargs)
    dt = time.time() - t0
    in_len = input_tensors["input_ids"].shape[-1]
    gen_ids = out[0][in_len:]  # Decore only the answer, not the full prompt.
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    out_len = out[0].shape[-1] - in_len
    return in_len, out_len, dt, output_text


def run_sanity_inference(tokenizer, mod, user_prompt: str):
    model_ctx = getattr(mod.config, "n_ctx", None) or getattr(mod.config, "max_position_embeddings", 1024)
    max_new_tokens = 16
    safety = 8
    max_input_tokens = min(256, model_ctx - max_new_tokens - safety)
    input_tensors = tokenizer(
        user_prompt,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt"
    ).to(mod.device)
    gen_kwargs = dict(
        **input_tensors,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        use_cache=True
    )
    t0 = time.time()
    with torch.no_grad():
        out = mod.generate(**gen_kwargs)
    dt = time.time() - t0
    in_len = input_tensors["input_ids"].shape[-1]
    out_len = out[0].shape[-1] - in_len
    output_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return in_len, out_len, dt, output_text


def supports_chat(tokenizer):
    return hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None


def run_semantic_analysis(tokenizer, mod, sa_user_prompt: str):
    """
    Runs the semantic analyzer prompt on the free-form detection_text
    and returns (input_tokens, output_tokens, latency_s, sa_reply_text).
    """
    in_len, out_len, secs, sa_text = run_one_inference(tokenizer, mod, system_prompt=ROLE_SA,
                                                       user_prompt=sa_user_prompt,
                                                       max_new_tokens=SA_MAX_NEW_TOKENS, temperature=0.0, top_p=1.0
                                                       )
    return in_len, out_len, secs, sa_text.strip()


def last_energy_kwh(csv_path):
    """
    Return energy_kwh from the last row of a CodeCarbon emissions.csv file.
    Returns None if missing or malformed.
    """
    try:
        with open(csv_path, encoding="utf-8") as csv_file:
            lines = [ln.strip() for ln in csv_file if ln.strip()]
    except (FileNotFoundError, OSError):
        return None
    if len(lines) <= 1:
        return None
    header = lines[0].split(",")
    last_line = lines[-1].split(",")
    try:
        k_idx = header.index("energy_consumed")
        return float(last_line[k_idx])
    except (ValueError, IndexError):
        return None


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


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--dataset", required=True, help="Path to smartbugs-curated root.")
    ap.add_argument("--prompt", required=True, choices=sorted(PROMPT_TEMPLATES.keys()))
    ap.add_argument("--persona", action="store_true", help="Enable VD persona via system prompt.")
    ap.add_argument("--strip_comments", action="store_true", default=True, dest="strip_comments")
    ap.add_argument("--resume_json", help="Optional JSON to resume and skip processed files")
    ap.add_argument("--sa_prompt", default="SA", choices=sorted(SA_PROMPT_TEMPLATES_MAP.keys()))
    args = ap.parse_args()
    # The model name uses "/" which would create a subfolder and hence it's replaced by "_".
    safe_model_name = args.model.replace("/", "__")

    # Derive effective prompt key (explicitly encode persona)
    if args.persona:
        effective_prompt = f"{args.prompt}_PERSONA"
    else:
        effective_prompt = args.prompt
    base_dir = Path("results") / safe_model_name / effective_prompt
    base_dir.mkdir(parents=True, exist_ok=True)

    run_id = len(list(base_dir.glob("run_*.json"))) + 1
    run_tag = f"run_{run_id:03d}"

    csv_path = base_dir / f"{run_tag}.csv"
    json_path = base_dir / f"{run_tag}.json"

    print(f"\nLoading model: {args.model}")
    tokenizer, model = load_model(args.model)
    # Warm-up to stabilize clock/caching.
    print("Running warm-up inference...")
    _ = run_one_inference(tokenizer, model, None, "Warm up.",
                          max_new_tokens=16, temperature=0.0, top_p=1.0)
    # Dedicated directory for CodeCarbon tracker for this model.
    out_dir = base_dir / ".codecarbon"
    out_dir.mkdir(exist_ok=True)
    # CSV.
    file_output = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(file_output)
    # HEADERS.
    writer.writerow([
        "category", "file_name",
        "vd_input_tokens", "vd_output_tokens", "vd_total_tokens",
        "vd_latency_s", "vd_energy_kwh", "vd_emissions_kg",

        "sa_input_tokens", "sa_output_tokens", "sa_total_tokens",
        "sa_latency_s", "sa_energy_kwh", "sa_emissions_kg",

        "vd_prompt", "vd_reply", "sa_prompt", "sa_reply"
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
    if args.persona:
        sys_p = ROLE_VD.strip()
    else:
        sys_p = None
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
            vd_prompt = get_prompt(tpl, code)

            # Emptying the PyTorch CUDA cache so that each inference starts with a clean state (helps with OOM too).
            # It doesn't free the VRAM of the model (good).
            # Constant overhead (marginal).
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Per-inference Tracking.
            vd_tracker = EmissionsTracker(
                measure_power_secs=10,
                output_dir=out_dir,
                save_to_file=True,
                project_name=safe_model_name,
                experiment_id=f"{cat_key}/{file_name}",
                log_level="error"
            )

            sa_tracker = EmissionsTracker(
                measure_power_secs=10,
                output_dir=out_dir,
                save_to_file=True,
                project_name=safe_model_name,
                experiment_id=f"SA_{cat_key}/{file_name}",
                log_level="error"
            )

            # Right now, the CodeCarbon tracker still measures only the detection step, not the SA call.
            # That’s fine for the moment if your priority is wiring the semantics.
            # If later you want energy of the whole pipeline (detection + SA), just move tracker.stop()
            # to after run_semantic_analysis so it wraps both calls.
            vd_tracker.start()
            vd_in_t, vd_out_t, vd_secs, vd_reply = run_one_inference(tokenizer, model, sys_p, vd_prompt,
                                                            max_new_tokens=VD_MAX_NEW_TOKENS, temperature=TEMPERATURE,
                                                            top_p=TOP_P)
            vd_emissions_kg = vd_tracker.stop()  # Per-inference kg CO2e.
            # Read energy_kwh for the last appended row.
            vd_energy_kwh = last_energy_kwh(out_dir / "emissions.csv")
            # Helps with OOM errors.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # --- Semantic analysis step (second prompt) ---
            sa_prompt = get_prompt(sa_template, vd_reply)
            sa_tracker.start()
            sa_in_t, sa_out_t, sa_secs, sa_reply = run_semantic_analysis(
                tokenizer,
                model,
                sa_prompt
            )
            sa_emissions_kg = sa_tracker.stop()
            sa_energy_kwh = last_energy_kwh(out_dir / "emissions.csv")
            # Parse semantic analyzer output → dict ('access_control': 0/1, ...)
            prediction_map = parse_sa_output(sa_reply)
            # Build JSON result entry.
            json_results.append({
                "model_name": args.model,
                "prompt_key": effective_prompt,
                "category": cat_name,
                "file_name": file_name,
                "detection_output": vd_reply,
                "semantic_output": sa_reply,
                "prediction_map": prediction_map,  # 0/1 dictionary.
            })
            # CSV row.
            writer.writerow([
                cat_name, file_name,

                vd_in_t, vd_out_t, vd_in_t + vd_out_t,
                f"{vd_secs:.6f}",
                f"{vd_energy_kwh:.9f}" if vd_energy_kwh is not None else "",
                f"{vd_emissions_kg:.9f}" if vd_emissions_kg is not None else "",

                sa_in_t, sa_out_t, sa_in_t + sa_out_t,
                f"{sa_secs:.6f}",
                f"{sa_energy_kwh:.9f}" if sa_energy_kwh is not None else "",
                f"{sa_emissions_kg:.9f}" if sa_emissions_kg is not None else "",

                vd_prompt, vd_reply, sa_prompt, sa_reply
            ])
            file_output.flush()  # Ensure rows land even if interrupted.
    file_output.close()
    # ---- Save one JSON per run ----
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_results, jf, indent=2)
    print(f"Saved JSON results: {json_path}")
    print(f"Saved CSV results: {csv_path}")


if __name__ == "__main__":
    main()

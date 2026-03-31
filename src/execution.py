# Given a model, a dataset, and a prompt key (prompt pattern/template to use),
# this script automatically runs a complete LLM-based vulnerability-detection experiment
# and measures its computational footprint.
# The output is a directory for the given prompt, which contains a CSV for each run.
# Each CSV has stats about each contract's inference.

# After VD, the output is converted into a binary prediction map
# either through a deterministic parser or through the Semantic Analysis prompt.
# The output is a JSON with, among other stats, the prediction map (for quality evaluation).

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch, time, csv
import json
import re
import argparse
from pathlib import Path
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM
from difflib import get_close_matches
from .vulnerabilities_constants import CATEGORIES, KEYS_TO_CATEGORIES
from .prompts import (
    # Include the vulnerability-detection templates.
    ORIGINAL_ZS, ORIGINAL_ZS_COT, ZS, ZS_COT, FS, ROLE_VD,
    # Include also the Semantic Analysis templates.
    SA, ROLE_SA,
)

# If GPU CUDA is available, then use bfloat16 (b stands for Brain in Google Brain), otherwise use float32.
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
VD_MAX_NEW_TOKENS = int(os.getenv("VD_MAX_NEW_TOKENS", "2048"))
SA_MAX_NEW_TOKENS = int(os.getenv("SA_MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "1"))

PROMPT_TEMPLATES = {
    "ORIGINAL_ZS": ORIGINAL_ZS,
    "ORIGINAL_ZS_COT": ORIGINAL_ZS_COT,
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
    tok.pad_token = tok.eos_token
    mdl = (AutoModelForCausalLM.from_pretrained(
        model_name, dtype=DTYPE,
        device_map="auto"
        # device_map=infer_auto_device_map(mdl)
        # device_map={"": 1}
    ).eval())
    return tok, mdl


def get_prompt(prompt_template: str, code: str) -> str:
    # The templates use "{input}".
    return prompt_template.replace("{input}", code)


def run_one_inference(tokenizer, mod, system_prompt, user_prompt, max_new_tokens, temperature, top_p):
    if supports_chat(tokenizer):
        return run_chat_inference(tokenizer, mod, system_prompt, user_prompt, max_new_tokens, temperature, top_p)
    else:  # Fallback to plain text generation when the model doesn’t support a chat format.
        return run_sanity_inference(tokenizer, mod, user_prompt, max_new_tokens, temperature, top_p)


def run_chat_inference(tokenizer, mod, system_prompt: str | None, user_prompt: str, max_new_tokens, temperature, top_p):
    input_tensors = tokenize_chat_input(tokenizer, mod, system_prompt, user_prompt)
    return generate_from_inputs(tokenizer, mod, input_tensors, max_new_tokens, temperature, top_p)


def run_sanity_inference(tokenizer, mod, user_prompt: str, max_new_tokens, temperature, top_p):
    model_ctx = getattr(mod.config, "n_ctx", None) or getattr(mod.config, "max_position_embeddings", 1024)
    safety = 8
    max_input_tokens = min(256, model_ctx - max_new_tokens - safety)
    input_tensors = tokenize_plain_input(tokenizer, mod, user_prompt, max_input_tokens)
    return generate_from_inputs(tokenizer, mod, input_tensors, max_new_tokens, temperature, top_p)


def tokenize_chat_input(tokenizer, mod, system_prompt: str | None, user_prompt: str):
    msgs = []
    # System prompt: the instruction that defines the model’s role or behavior. Could be missing.
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    # Get the token ids and the attention mask.
    return tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(mod.device)


def tokenize_plain_input(tokenizer, mod, user_prompt: str, max_input_tokens: int):
    return tokenizer(
        user_prompt,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt"
    ).to(mod.device)


def generate_from_inputs(tokenizer, mod, input_tensors, max_new_tokens, temperature, top_p):
    # Generation settings. Sampling only if temperature > 0 (not all models support temperature).
    do_sample = temperature > 0
    gen_kwargs = dict(
        **input_tensors,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None
        gen_kwargs["top_k"] = None
    # Run generation without the grad, measure latency seconds.
    t0 = time.time()
    with torch.inference_mode():
        # outputs = mod(**input_tensors, use_cache=True)
        # next_token_scores = outputs.logits[:, -1, :]
        # print("logits has_nan:", torch.isnan(next_token_scores).any().item(), flush=True)
        # print("logits has_inf:", torch.isinf(next_token_scores).any().item(), flush=True)
        # print(f"logits min: {next_token_scores.min().item():.12e}", flush=True)
        # print(f"logits max: {next_token_scores.max().item():.12e}", flush=True)
        # print(f"logits mean: {next_token_scores.mean().item():.12e}", flush=True)
        # probs = torch.softmax(next_token_scores, dim=-1)

        # Offloaded to save memory (as it goes into OOM).
        # Check: https://huggingface.co/docs/transformers/en/kv_cache
        # if mod.device.type == "cuda":
        # gen_kwargs["cache_implementation"] = "offloaded"
        out = mod.generate(**gen_kwargs)
    dt = time.time() - t0

    in_len = input_tensors["input_ids"].shape[-1]
    gen_ids = out[0][in_len:]  # Decode only the answer, not the full prompt.
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    out_len = out[0].shape[-1] - in_len
    return in_len, out_len, dt, output_text


def supports_chat(tokenizer):
    return hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None


# ------------ OUTPUT NORMALIZATION / PARSING. ------------
# Map lowercase canonical names → proper category names.
CANONICAL = {name.lower(): name for name in CATEGORIES}

# Add synonym normalization
SYNONYMS = {
    "short address attack": "Short Addresses",
    "short address": "Short Addresses",
    "arithmetic issues": "Arithmetic",
    "integer overflow": "Arithmetic",
    "integer underflow": "Arithmetic",
    "unchecked return values": "Unchecked Low Level Calls",
    "unchecked return values for low level calls": "Unchecked Low Level Calls",
    "dos": "Denial Of Service",
    "transaction ordering dependence": "Front Running",
    "timestamp dependence": "Time Manipulation",
}


def normalize_name(s):
    s = s.lower().strip()
    # Remove angle brackets.
    s = s.replace("<", " ").replace(">", " ")
    # Strip Markdown and bullets.
    s = re.sub(r"[*_`#>\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Strip leading "id:".
    s = re.sub(r"^id\s*:\s*", "", s).strip()
    # Strip numbering like "1. Foo" or "1) Foo".
    s = re.sub(r"^\d+\s*[.)]\s*", "", s)
    # Remove parenthetical notes.
    s = re.sub(r"\([^)]*\)", "", s).strip()
    # Direct synonyms.
    if s in SYNONYMS:
        return SYNONYMS[s]
    # Exact canonical match.
    if s in CANONICAL:
        return CANONICAL[s]
    # Fuzzy match to canonical names.
    match = get_close_matches(s, CANONICAL.keys(), n=1, cutoff=0.75)
    if match:
        return CANONICAL[match[0]]
    return None


def normalize_vd_verdict(s):
    s = s.lower().strip()
    s = re.sub(r"[^a-z ]+", "", s)
    if s == "present":
        return 1
    if s == "absent":
        return 0
    if s == "uncertain":
        return 0
    return None


def parse_vd_output(vd_text: str):
    """
    Deterministically parse VD output and return:
    - prediction_map
    - parsed_labels

    Supported patterns:
    1. Access Control: Present
    2. <ID: Access Control>: Present
    3. ID: Access Control
       Explanation: Present
    """
    prediction_map = {name: 0 for name in CATEGORIES}
    parsed_labels = set()

    lines = [line.strip() for line in vd_text.splitlines() if line.strip()]

    current_label = None

    for line in lines:
        # Pattern 1 / 2: same-line verdict, e.g.
        # "Access Control: Present"
        # "<ID: Access Control>: Present"
        if ":" in line:
            left, right = line.rsplit(":", 1)
            raw_name = left.strip()
            raw_verdict = right.strip()

            norm_name = normalize_name(raw_name)

            if norm_name is not None:
                norm_verdict = extract_vd_verdict(raw_verdict)
                if norm_verdict is not None:
                    prediction_map[norm_name] = norm_verdict
                    parsed_labels.add(norm_name)
                    current_label = None
                    continue

        # Pattern 3a: "ID: Access Control"
        m_id = re.match(r"^ID\s*:\s*(.+)$", line, flags=re.IGNORECASE)
        if m_id:
            current_label = normalize_name(m_id.group(1))
            continue

        # Pattern 3b: "Explanation: Present"
        if current_label is not None:
            m_expl = re.match(r"^Explanation\s*:\s*(.+)$", line, flags=re.IGNORECASE)
            if m_expl:
                norm_verdict = extract_vd_verdict(m_expl.group(1))
                if norm_verdict is not None:
                    prediction_map[current_label] = norm_verdict
                    parsed_labels.add(current_label)
                current_label = None
                continue

    return prediction_map, parsed_labels


def extract_vd_verdict(text: str):
    text = text.strip()
    text = re.sub(r"[*_`]+", "", text)  # remove markdown emphasis
    text = text.strip()
    # Reject copied templates like "Present | Absent | Uncertain"
    if "|" in text:
        return None
    m = re.match(r"^(Present|Absent|Uncertain)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    return normalize_vd_verdict(m.group(1))


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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--dataset", required=True, help="Path to smartbugs-curated root.")
    ap.add_argument("--prompt", required=True, choices=sorted(PROMPT_TEMPLATES.keys()))
    ap.add_argument("--role", action="store_true", help="Enable VD role pattern via system prompt.")
    ap.add_argument("--no_strip_comments", action="store_false", dest="strip_comments")
    ap.set_defaults(strip_comments=True)
    ap.add_argument("--parser_mode",
        default="deterministic",
        choices=["deterministic", "sa"],
        help="How to convert VD output into the final binary prediction map."
    )
    ap.add_argument("--sa_prompt", default="SA", choices=sorted(SA_PROMPT_TEMPLATES_MAP.keys()))
    return ap.parse_args()


def prepare_run_paths(model_name: str, effective_prompt: str):
    # The model name uses "/", which would create a subfolder, and hence it's replaced by "_".
    safe_model_name = model_name.replace("/", "__")
    base_dir = Path("results") / safe_model_name / effective_prompt
    base_dir.mkdir(parents=True, exist_ok=True)

    run_id = len(list(base_dir.glob("run_*.json"))) + 1
    run_tag = f"run_{run_id:03d}"

    csv_path = base_dir / f"{run_tag}.csv"
    json_path = base_dir / f"{run_tag}.json"
    # Dedicated directory for CodeCarbon tracker for this model.
    out_dir = base_dir / ".codecarbon"
    out_dir.mkdir(exist_ok=True)

    return safe_model_name, csv_path, json_path, out_dir


# ---------- main ----------
def main():
    args = parse_args()
    # Derive an effective prompt key (explicitly encode the role).
    if args.role:
        effective_prompt = f"{args.prompt}_ROLE"
    else:
        effective_prompt = args.prompt
    effective_prompt = f"{effective_prompt}_{args.parser_mode.upper()}"
    safe_model_name, csv_path, json_path, out_dir = prepare_run_paths(args.model, effective_prompt)
    tracker = EmissionsTracker(measure_power_secs=1, output_dir=str(out_dir), save_to_file=True,
                               project_name=safe_model_name, experiment_id="full_run", log_level="error")
    file_output = None
    try:
        print(f"\nLoading model: {args.model}")
        tokenizer, model = load_model(args.model)
        # Warm-up to stabilize clock/caching.
        print("Running warm-up inference...")
        _ = run_one_inference(tokenizer, model, None, "Warm up.",
                              max_new_tokens=16, temperature=TEMPERATURE, top_p=TOP_P)
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
        tpl = PROMPT_TEMPLATES[args.prompt]
        sa_template = SA_PROMPT_TEMPLATES_MAP[args.sa_prompt]
        if args.role:
            sys_p = ROLE_VD.strip()
        else:
            sys_p = None
        json_results = []
        for cat_key, cat_name in KEYS_TO_CATEGORIES.items():
            category_directory = os.path.join(args.dataset, str(cat_key))
            print(f"Analyzing files of vulnerability category: {cat_name}")
            if not os.path.isdir(category_directory):
                continue
            for file_name in sorted(os.listdir(category_directory)):
                if not file_name.endswith(".sol"):
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
                    torch.cuda.reset_peak_memory_stats()

                print(f"Executing detection for {file_name}")
                vd_task_name = f"vd_{cat_key}_{file_name}"

                tracker.start_task(vd_task_name)
                vd_in_t, vd_out_t, vd_secs, vd_reply = run_one_inference(tokenizer, model, sys_p, vd_prompt,
                                                                         max_new_tokens=VD_MAX_NEW_TOKENS,
                                                                         temperature=TEMPERATURE,
                                                                         top_p=TOP_P)
                vd_emission_data = tracker.stop_task()

                print(f"Completed detection for {file_name} "
                      f"(in={vd_in_t}, out={vd_out_t}, time={vd_secs:.2f}s)"
                      )
                vd_energy_kwh = vd_emission_data.energy_consumed
                vd_emissions_kg = vd_emission_data.emissions
                # Helps with OOM errors.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # --- Convert VD output into the final prediction map ---
                if args.parser_mode == "deterministic":
                    prediction_map, parsed_labels = parse_vd_output(vd_reply)
                    if len(parsed_labels) < len(CATEGORIES):
                        print(f"\n[DEBUG] Deterministic parser parsed {len(parsed_labels)}/{len(CATEGORIES)} labels")
                        print(vd_reply[:1000], flush=True)
                    sa_in_t = 0
                    sa_out_t = 0
                    sa_secs = 0.0
                    sa_energy_kwh = 0.0
                    sa_emissions_kg = 0.0
                    sa_prompt = ""
                    sa_reply = ""

                elif args.parser_mode == "sa":
                    sa_prompt = get_prompt(sa_template, vd_reply)
                    sa_task_name = f"sa_{cat_key}_{file_name}"
                    tracker.start_task(sa_task_name)
                    sa_in_t, sa_out_t, sa_secs, sa_reply = run_semantic_analysis(
                        tokenizer,
                        model,
                        sa_prompt
                    )
                    sa_emission_data = tracker.stop_task()
                    sa_energy_kwh = sa_emission_data.energy_consumed
                    sa_emissions_kg = sa_emission_data.emissions
                    # Parse semantic analyzer output → dict ('access_control': 0/1, ...)
                    prediction_map = parse_sa_output(sa_reply)
                # Build JSON result entry.
                json_results.append({
                    "model_name": args.model,
                    "prompt_key": effective_prompt,
                    "parser_mode": args.parser_mode,
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
        # ---- Save one JSON per run ----
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_results, jf, indent=2)
        print(f"Saved JSON results: {json_path}")
        print(f"Saved CSV results: {csv_path}")
    finally:
        if file_output is not None and not file_output.closed:
            file_output.close()
        tracker.stop()


if __name__ == "__main__":
    main()

from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time, csv, os, pathlib
import json
import re
import argparse

from prompts import (
    PROMPT_VD_VARIANT_1, PROMPT_VD_VARIANT_2, PROMPT_VD_VARIANT_3,
    ORIGINAL_PROMPT_VD, ORIGINAL_PROMPT_VD_RP,
    PROMPT_VD_FEW_SHOTS, PROMPT_VD_FEW_SHOTS_1, PROMPT_VD_FEW_SHOTS_2, PROMPT_VD_FEW_SHOTS_3
)

# ---------- config ----------
# If GPU CUDA available then use bfloat16 (b stands for Brain in Google Brain), else float32.
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "4000")) # Just a protection against endless/degenerate loops.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))

PROMPT_MAP = {
    "RP": ORIGINAL_PROMPT_VD_RP,
    "ORIGINAL": ORIGINAL_PROMPT_VD,
    "VARIANT_1": PROMPT_VD_VARIANT_1,
    "VARIANT_2": PROMPT_VD_VARIANT_2,
    "VARIANT_3": PROMPT_VD_VARIANT_3,
    "FEWSHOTS": PROMPT_VD_FEW_SHOTS,
    "FEWSHOTS_1": PROMPT_VD_FEW_SHOTS_1,
    "FEWSHOTS_2": PROMPT_VD_FEW_SHOTS_2,
    "FEWSHOTS_3": PROMPT_VD_FEW_SHOTS_3,
}

CATEGORIES = [
    # "access_control", "arithmetic", "bad_randomness", "denial_of_service",
    # "front_running", "reentrancy", "short_addresses", "time_manipulation",
    # "unchecked_low_level_calls"

    "front_running"
]

# ---------- utils ----------
def strip_solidity_comments(src: str) -> str:
    src = re.sub(r"/\*[\s\S]*?\*/", "", src)     # Block comments.
    src = re.sub(r"//.*", "", src)               # Line comments.
    src = re.sub(r"\n\s*\n+", "\n\n", src)
    return src.strip()


# Prompts.
with open("prompts.txt", encoding="utf-8") as f:
    # Read all non-empty lines from a file f, removes extra spaces, and stores them in a list.
    PROMPTS = [p.strip() for p in f if p.strip()]


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
    # doesn't matter but it might show warnings).
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

def last_emissions_row(csv_path):
    """
    Return (energy_kwh, emissions_kg) from the last row of emissions.csv of CodeCarbon.
    """
    try:
        with open(csv_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) <= 1:
            return None, None
        last_line_values = lines[-1].split(",")
        headers = lines[0].split(",")
        # CodeCarbon tipicamente: timestamp,project,run_id,experiment_id,os,python,codecarbon_version,cpu_count,cpu_model,ram,
        # gpu_count,gpu_model,longitude,latitude,region,country,cloud_provider,cloud_region,emissions,emissions_rate,energy_consumed,energy_consumed_unit,
        # duration,ram_total_size,cpu_usage, ... (può variare)
        # Cerchiamo per nome colonna? Manteniamolo semplice: emissions = last_line_values[idx_e], energy_consumed = last_line_values[idx_kwh]
        e_idx = headers.index("emissions") if "emissions" in headers else None
        k_idx = headers.index("energy_consumed") if "energy_consumed" in headers else None
        emissions_kg = float(last_line_values[e_idx]) if e_idx is not None else None
        energy_kwh = float(last_line_values[k_idx]) if k_idx is not None else None
        return energy_kwh, emissions_kg
    except Exception:
        return None, None


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--dataset", required=True, help="Path to smartbugs-curated root")
    ap.add_argument("--prompt", required=True, choices=sorted(PROMPT_MAP.keys()))
    ap.add_argument("--out_csv", default="results.csv")
    ap.add_argument("--system", default="You are a vulnerability detector for a smart contract.")
    ap.add_argument("--strip_comments", action="store_true")
    ap.add_argument("--resume_json", help="Optional JSON to resume and skip processed files")
    args = ap.parse_args()
    print(f"\nLoading model: {args.model}")
    tokenizer, model = load_model(args.model)
    # Warm-up to stabilize clock/caching.
    print("Running warm-up inference...")
    _ = run_one_inference(tokenizer, model, None,"Warm up.")
    # Dedicated directory for CodeCarbon tracker for this model.
    safe_m = args.model.replace("/", "__")
    out_dir = f".codecarbon_{safe_m}"
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    # CSV.
    new_file = not pathlib.Path(args.out_csv).exists()
    fout = open(args.out_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if new_file:
        # HEADERS.
        writer.writerow([
            "model_name", "prompt_key", "category", "file_name",
            "input_tokens", "output_tokens", "total_tokens",
            "latency_s", "energy_kwh_total", "emissions_kg_total",
            "energy_kwh_per_sample", "emissions_kg_per_sample",
            "reply"
        ])
    processed = set()
    if args.resume_json and pathlib.Path(args.resume_json).exists():
        with open(args.resume_json, encoding="utf-8") as f:
            try:
                prev = json.load(f)
                processed = {(x.get("category"), x.get("file_name")) for x in prev}
            except Exception:
                processed = set()
    tpl = PROMPT_MAP[args.prompt]
    sys_p = args.system.strip() if args.system else None
    for cat in CATEGORIES:
        category_directory = os.path.join(args.dataset, cat)
        print(f"Analyzing files of vulnerability category: {cat}")
        if not os.path.isdir(category_directory):
            continue
        for file_name in os.listdir(category_directory):
            if not file_name.endswith(".sol"):
                continue
            key = (cat, file_name)
            if key in processed:
                continue
            with open(os.path.join(category_directory, file_name), encoding="utf-8") as f:
                code = f.read()
            if args.strip_comments:
                code = strip_solidity_comments(code)
            user_prompt = get_prompt(tpl, code)
            # ------- Per-inference Tracking -------
            tracker = EmissionsTracker(
                measure_power_secs=1,
                output_dir=out_dir,
                save_to_file=True,
                project_name=safe_m,
                experiment_id=f"{cat}/{file_name}"  # Helps identify rows in emissions.csv
            )
            tracker.start()
            in_t, out_t, secs, text = run_one_inference(tokenizer, model, sys_p, user_prompt)
            emissions_kg_this = tracker.stop()  # Per-inference kg CO2e.
            # Read energy_kwh for *this* row (the last appended row).
            energy_kwh_this, emissions_kg_csv = last_emissions_row(pathlib.Path(out_dir, "emissions.csv"))
            if emissions_kg_csv is not None:
                emissions_kg_this = emissions_kg_csv
            # -------------------------------------

            writer.writerow([
                args.model, args.prompt, cat, file_name,
                in_t, out_t, in_t + out_t,
                f"{secs:.6f}",
                f"{energy_kwh_this:.9f}" if energy_kwh_this is not None else "",
                f"{emissions_kg_this:.9f}" if emissions_kg_this is not None else "",
                "", "",  # per-sample already recorded, so leave “per_sample” cols empty or duplicate
                text
            ])
            fout.flush()  # Ensure rows land even if interrupted.
    fout.close()
    print(f"Done -> {args.out_csv}")

if __name__ == "__main__":
    main()



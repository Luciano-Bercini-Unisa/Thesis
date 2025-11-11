from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time, csv, os, pathlib

# Configuration.
MODELS = [
    "microsoft/Phi-3.5-mini-instruct",
]
PROMPT_TYPE = os.getenv("PROMPT_TYPE", "zero_shot")
# If GPU available then use bfloat16 (b stands for Brain in Google Brain), else float32.
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
TOP_P = float(os.getenv("TOP_P", "1.0"))

# Prompts.
with open("prompts.txt", encoding="utf-8") as f:
    # Read all non-empty lines from a file f, removes extra spaces, and stores them in a list.
    PROMPTS = [p.strip() for p in f if p.strip()]

# CSV output.
OUT_CSV = "results.csv"
headers = [
    "model_name", "prompt_type", "prompt_id",
    "input_tokens", "output_tokens", "total_tokens",
    "latency_s",
    "energy_kwh_total", "emissions_kg_total",
    "energy_kwh_per_prompt", "emissions_kg_per_prompt",
    "reply"
]

new_file = not pathlib.Path(OUT_CSV).exists()
write_file = open(OUT_CSV, "a", newline="", encoding="utf-8")
writer = csv.writer(write_file)
if new_file:
    writer.writerow(headers)


def run_one_inference(tokenizer, mod, prompt):
    chat = [{"role": "user", "content": prompt}]
    # Get the token ids and the attention mask.
    input_tensors = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors="pt", padding=True).to(mod.device)
    # Generation settings. Sampling only if temperature > 0 (not all models support temperature; doesn't matter but it might show warnings).
    output_generation_settings = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=(TEMPERATURE>0), temperature=TEMPERATURE, top_p=TOP_P)
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


def load_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=DTYPE, device_map="auto").eval()
    return tok, model


def last_emissions_row(csv_path):
    """
    Return (energy_kwh, emissions_kg) from the last row of emissions.csv of CodeCarbon.
    """
    try:
        with open(csv_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) <= 1:
            return None, None
        cols = lines[-1].split(",")
        # CodeCarbon tipicamente: timestamp,project,run_id,experiment_id,os,python,codecarbon_version,cpu_count,cpu_model,ram,
        # gpu_count,gpu_model,longitude,latitude,region,country,cloud_provider,cloud_region,emissions,emissions_rate,energy_consumed,energy_consumed_unit,
        # duration,ram_total_size,cpu_usage, ... (può variare)
        # Cerchiamo per nome colonna? Manteniamolo semplice: emissions = cols[idx_e], energy_consumed = cols[idx_kwh]
        header_cols = lines[0].split(",")
        e_idx = header_cols.index("emissions") if "emissions" in header_cols else None
        k_idx = header_cols.index("energy_consumed") if "energy_consumed" in header_cols else None
        emissions_kg = float(cols[e_idx]) if e_idx is not None else None
        energy_kwh = float(cols[k_idx]) if k_idx is not None else None
        return energy_kwh, emissions_kg
    except Exception:
        return None, None


for mod in MODELS:
    print(f"\nLoading model: {mod}")
    tokenizer, model = load_model(mod)
    # Warm-up to stabilize clock/caching.
    print("Running warm-up inference...")
    _ = run_one_inference(tokenizer, model, "Warm up.", )
    print("Starting emission tracker...")
    # Tracker CodeCarbon for this model (dedicated directory).
    safe_m = mod.replace("/", "__")
    out_dir = f".codecarbon_{safe_m}"
    pathlib.Path(out_dir).mkdir(exist_ok=True)
    tracker = EmissionsTracker(measure_power_secs=1, output_dir=out_dir, save_to_file=True)
    tracker.start()

    rows_tmp = []
    for i, p in enumerate(PROMPTS):
        in_t, out_t, secs, text = run_one_inference(tokenizer, model, p)
        print(f"\n[{mod}] Prompt {i} | in:{in_t} out:{out_t} tok | {secs:.2f}s")
        rows_tmp.append((i, in_t, out_t, in_t+out_t, secs, text))

    emissions_kg_total = tracker.stop()  # kg CO2e
    # Try to obtain kWh from CSV of CodeCarbon.
    energy_kwh_total, emissions_kg_cc = last_emissions_row(pathlib.Path(out_dir, "emissions.csv"))
    if emissions_kg_cc is not None:
        emissions_kg_total = emissions_kg_cc

    n = max(1, len(rows_tmp))
    per_kwh = (energy_kwh_total / n) if energy_kwh_total is not None else None
    per_kg  = (emissions_kg_total / n) if emissions_kg_total is not None else None

    # Scrivi righe nel CSV aggregato
    for (pid, in_t, out_t, tot_t, secs, text) in rows_tmp:
        writer.writerow([
            mod, PROMPT_TYPE, pid,
            in_t, out_t, tot_t,
            f"{secs:.6f}",
            f"{energy_kwh_total:.9f}" if energy_kwh_total is not None else "",
            f"{emissions_kg_total:.9f}" if emissions_kg_total is not None else "",
            f"{per_kwh:.9f}" if per_kwh is not None else "",
            f"{per_kg:.9f}" if per_kg is not None else "",
            text
        ])

write_file.close()
print(f"\nDone -> {OUT_CSV}")

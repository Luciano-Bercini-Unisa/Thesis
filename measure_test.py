from codecarbon import EmissionsTracker
# Causal LM = a model trained to predict the next token in a sequence, one token at a time, based only on everything before it (GPT, LLaMA, Mistral, Phi...).
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, time, csv, os

MODEL = os.getenv("MODEL", "microsoft/Phi-3.5-mini-instruct")
# Use bfloat16 on GPU for speed/memory; else float32 on CPU.
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Downloads/loads the tokenizer for the model.
tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
# Downloads/loads the model weights, places them on available device(s) automatically, sets evaluation mode (i.e. inference mode. In PyTorch, every neural network has two modes: Training mode and Evaluation Mode).
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=DTYPE, device_map="auto"
).eval()

# Build a chat-style prompt (max new tokens is only for output).
def run_one(prompt, max_new_tokens=1000, temperature=0.0, top_p=1.0):
    chat = [{"role":"user","content":prompt}]
    # Formats the chat history into the correct text form for that model as each model expects chat text in a specific style (apply_chat_template).
    # add_generation_prompt=True means pretty much "now it's your turn to reply".
    # return_tensors="pt" (output PyTorch tensors instead of plain lists, so the result is ready for computation on GPU/CPU).
    # padding=True means pad shorter sequences with zeros and create an attention mask (it ensures the model knows which tokens are real and which are just padding (to avoid confusion)).
    # Here we're running one prompt at a time so it doesn't matter as we don't shape-align for performance (batch size = 1), 
    # we just do it for the attention mask (padding might still be added for efficiency even with batch=1).

    # to(model.device) moves the data to either GPU or keeps them in CPU.
    # After this line, inputs becomes a Python dictionary with two tensors inside:
    # "input_ids": tensor([[128001, 1629, 2507, 19389, 3050, 128002]]),
    # "attention_mask": tensor([[1, 1, 1, 1, 1, 1]])
    inputs = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt", padding=True).to(model.device)
    # Number of input tokens.
    in_len = inputs.shape[-1]
    # Generation settings. Sampling only if temperature > 0 (not all models support temperature; doesn't matter but it might show warnings).
    cfg = dict(max_new_tokens=max_new_tokens, do_sample=(temperature>0), temperature=temperature, top_p=top_p)
    # Run generation without grad, measure latency seconds.
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(inputs, **cfg)
    dt = time.time() - t0
    # Slice only the newly generated tokens; return input count, output count, and latency.
    out_ids = out[0][in_len:]

    text = tok.decode(out_ids, skip_special_tokens=True)
    return in_len, out_ids.numel(), dt, text

# One short run to stabilize clocks/caches so later timings are fair.
_ = run_one("Warm up.", max_new_tokens=10)

# Load prompts (read non-empty lines from prompts.txt into a list).
prompts = [p.strip() for p in open("prompts.txt", encoding="utf-8") if p.strip()]

# Poll power every 1 s.
tracker = EmissionsTracker(measure_power_secs=1)
# Begin logging energy/CO2e.
tracker.start()

# For each prompt, run the model, collect input tokens, output tokens, and latency. Store a row.
rows=[]
for i,p in enumerate(prompts):
    inp, outp, secs, text = run_one(p)
    print(f"\nPrompt {i}: {p}")
    print(f"Reply: {text}\n")
    rows.append([i, inp, outp, secs, text])

# Stop logging. Returns total kg CO2e for the whole batch.
emissions_kg = tracker.stop()

# Create results.csv. Write headers. Split the total emissions evenly across prompts as a simple per-prompt figure. Write one line per prompt with counts, latency, and per-prompt emissions.
with open("results.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["prompt_id", "input_tokens", "output_tokens", "latency_s", "reply", "emissions_kg"])
    per = emissions_kg/len(rows) if rows else 0.0
    for r in rows:
        w.writerow(r+[per])

print("Done -> results.csv")

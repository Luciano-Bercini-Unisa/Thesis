import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from transformers.utils import logging
from huggingface_hub import logging as hf_logging

logging.set_verbosity_info()
hf_logging.set_verbosity_info()

model_id = "HuggingFaceTB/SmolLM-135M"

print("Start")
t0 = time.time()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
print(f"Tokenizer loaded in {time.time() - t0:.1f}s")

t1 = time.time()
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=None,
                                             dtype=torch.float32,
                                             low_cpu_mem_usage=False)
print(f"Model loaded in {time.time() - t1:.1f}s")

print("Done")

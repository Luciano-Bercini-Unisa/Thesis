# Setup

1. python -m venv .venv  
   Create a private Python virtual environment.

2. venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/macOS)  
   Activate the virtual environment.

3. pip install -r requirements.txt  
   Install all required dependencies.

# Running experiment.py

This script is the single entry point for the full experimental pipeline:
1. Execution phase
   - Loads a chosen LLM.
   - Runs vulnerability-detection prompts on every Solidity contract in the SmartBugs-Curated dataset.
   - Runs a second Semantic Analysis (SA) prompt to produce structured predictions.
   - Measures inference latency, token usage, energy consumption, and CO₂ emissions (also through CodeCarbon).

2. Evaluation phase
   - Compares predictions against ground truth.
   - Computes per-class and macro-averaged metrics (precision, recall, specificity, F1-score).
   - Repeats experiments multiple times to mitigate model variance.

3. Aggregation phase
   - Combines quality metrics with energy and token statistics.
   - Produces consolidated reports at the model–prompt level.

## Outputs

For each run, the execution phase produces:
- A CSV file containing latency, token counts, energy usage, and emissions.
- A JSON file containing:
  - raw detection output,
  - semantic analysis output,
  - prediction_map used for quantitative evaluation.

Results are stored under:
results/<model_name>/<prompt_key>/

Examples:
results/Qwen__Qwen2.5-1.5B-Instruct/ZS/
results/Qwen__Qwen2.5-1.5B-Instruct/ZS_PERSONA/
results/Qwen__Qwen2.5-1.5B-Instruct/ZS_PERSONA_COT/

## Flags (execution phase)

--model MODEL_NAME        (default: microsoft/Phi-3.5-mini-instruct)
--dataset PATH            (required) Path to SmartBugs-Curated root
--prompt KEY              (required) One of: ZS, ZS_COT, FS
--persona                 Enable persona via system prompt
--strip_comments          Strip Solidity comments (enabled by default)
--resume_json FILE        Resume a previous run and skip processed files
--sa_prompt SA_KEY        Semantic analysis prompt (default: SA)

### Prompt keys

ZS       Zero-shot  
ZS_COT   Zero-shot + Chain-of-Thought  
FS       Few-shot  

When --persona is enabled, the effective prompt key is automatically suffixed with _PERSONA:
ZS_PERSONA  
ZS_PERSONA_COT

Persona conditioning is applied exclusively via the system prompt.

## Example models

sshleifer/tiny-gpt2                      Sanity-check model  
microsoft/Phi-3.5-mini-instruct          Small LLM  
Qwen/Qwen2.5-7B-Instruct                 Medium LLM  
deepseek-ai/deepseek-coder-33b-instruct  Large LLM  

# Example runs

## Zero-shot
python experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset smartbugs-curated/dataset_cleaned --prompt ZS

## Zero-shot + Persona
python experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset smartbugs-curated/dataset_cleaned --prompt ZS --persona

## Zero-shot + CoT + Persona
python experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset smartbugs-curated/dataset_cleaned --prompt ZS_COT --persona

## Few-shot
python experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset smartbugs-curated/dataset_cleaned --prompt FS


# Running evaluation.py

This script performs the evaluation phase only and should be run after execution has produced JSON outputs.

## What it does
- Loads ground-truth vulnerability annotations.
- Compares them against model predictions.
- Computes per-class and macro-averaged metrics (precision, recall, specificity, F1-score).

## Flags

--model MODEL_NAME  
--prompt PROMPT_KEY  

PROMPT_KEY must match the effective prompt key used during execution
(e.g. ZS, ZS_PERSONA, ZS_PERSONA_COT).

## Example
python evaluation.py --model Qwen__Qwen2.5-1.5B-Instruct --prompt ZS_PERSONA_COT

## Notes on reproducibility

- Persona conditioning is controlled exclusively via the --persona flag.
- Prompt keys explicitly encode the experimental condition.
- Each configuration is stored in a separate results folder, preventing accidental mixing of runs.
- The pipeline is fully deterministic except for model sampling, which is mitigated by repeated runs.

# Evaluation and aggregation (optional)

Normally, evaluation and aggregation are automatically executed by experiment.py.
The scripts below are provided for convenience when re-running analysis without
performing LLM inference again.

## Running evaluation.py

python evaluation.py --model Qwen__Qwen2.5-1.5B-Instruct --prompt ZS_PERSONA_COT

## Running aggregation.py

python aggregation.py

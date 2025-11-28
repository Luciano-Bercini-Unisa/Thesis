# Setup.
1. python -m venv .venv → Create your private Python sandbox.
2. venv\Scripts\activate → Enter that sandbox to run/install inside it.
3. pip install -r requirements.txt -> Recreate the sandbox using the requirements.

# Running measure_multi_test.py
This script executes a full vulnerability-detection experiment:
- Loads a chosen LLM.
- Runs detection prompts for every Solidity file in SmartBugs-Curated.
- Runs the second Semantic Analysis prompt.
- Computes energy and CO₂ emissions with CodeCarbon.
Saves two outputs:
- A CSV with latency, tokens, emissions.
- A JSON with detection text, semantic analysis text, and prediction_map (used for quality evaluation).

# 1. Flags.
--model MODEL_NAME            (default: microsoft/Phi-3.5-mini-instruct)
--dataset PATH                (required) Path to smartbugs-curated root
--prompt KEY                  (required) One of: ORIGINAL, STRUCTURED,
                                            VARIANT_1, VARIANT_2, VARIANT_3,
                                            FEW_SHOT, FEW_SHOT_WITH_EXPLANATION
--out_csv FILE                Output CSV path (auto-generated if omitted)
--system TEXT                 Optional system prompt
--no_strip_comments           Disable Solidity comment removal
--resume_json FILE            Resume a previous run and skip processed files
--sa_prompt SA_KEY            Semantic-analysis template (SA or STRUCTURED_SA)

# 2. Example models you can use.
- Qwen/Qwen2.5-1.5B-Instruct – extremely fast, good baseline.
- microsoft/Phi-3.5-mini-instruct – excellent quality/speed ratio.
- meta-llama/Meta-Llama-3-8B-Instruct – slower, stronger general reasoning.

# 3. Example: Basic run with ORIGINAL prompt.
python src/measure_multi_test.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dataset smartbugs-curated/dataset_cleaned \
    --prompt ORIGINAL

# Running evaluation_pipeline.py
This script performs the analysis stage:
- Extract ground truth.
- Compute perfect detections.
- Compute aggregated metrics (precision, recall, specificity, F1).

NOTE: Use it after you have JSON output from the LLM runs.

# 1. Flags.
--folder NAME      Name of the subfolder inside results/
                   Example: results/ORIGINAL/ → folder="ORIGINAL"

# 2. Example: Basic Run.
- python src/evaluation_pipeline.py --folder ORIGINAL
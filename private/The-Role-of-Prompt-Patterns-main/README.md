# Usage

## Step 1. Vulnerability detection. 
```bash
python3 main_vd.py --iteration NUM_ITERATION --prompt PROMPT --model MODEL --output FILENAME
```
Parameters:

* --iteration → Identifier of the current iteration (e.g., --iteration 1 for the first experiment with a given configuration).

* --prompt → Prompt to use (e.g., --prompt ORIGINAL uses the original prompt).

* --model → Model to use (e.g., --model gpt-4.1).

* --output → Output filename for the generated results.


## Step2. Semantic Analysis to Format the output. 
```bash
python3 main_sa.py --filename FILENAME
```
Provide the JSON file produced in Step 1.
This step reformats the raw results to follow a standardized output structure.


## Step 3. Metrics extraction.

To calculate metrics using the same method of the reference paper:
```bash
python3 eval_RP.py --filename PATH_TO_FILENAME
```

To calculate metrics using the scikit learn library:
```bash
python3 main_metrics2.py --filename PATH_TO_FILENAME
```

To calculate the metrics avarage across all iterations:
```bash
python3 metrics_avg.py --folderpath PATH_TO_FOLDER
```

# RQ1

## Method
We replicate the study of Chen et al. using the same prompts and the GPT-4 model.

## Results
The results are similar, and the study is reproducible.

# RQ2

## Method
We conducted an Ablation study on the original prompt. The original prompt template is:
#### ROLE + TASK + CoT
We identified 3 prompt variations:
* ROLE + TASK
* TASK + CoT
* TASK

## Results
Results confirm that these components make modest contributions to model reasoning and contextual understanding.

# RQ3 

## Method
We included a few-shot learning into the original prompt.

## Results
Few-shot learning strongly contributes to detection performance.


 # RQ4

 ## Method
 We used the prompts of previous RQ on a model with a reasoning mechanism.

 ## Results
 The reasoning model is more sensitive to prompt patterns than a model without a reasoning mechanism. CoT and Persona disturb the reasoning mechanism, while a simple task definition is more effective. 
set MODEL=Qwen/Qwen2.5-14B-Instruct
set DATASET=smartbugs-curated/dataset_cleaned_light

python -m src.experiment --model %MODEL% --dataset %DATASET% --prompt ORIGINAL_ZS_COT --role --runs 5
if errorlevel 1 pause

python -m src.experiment --model %MODEL% --dataset %DATASET% --prompt ZS_COT --role --runs 5
if errorlevel 1 pause

pause
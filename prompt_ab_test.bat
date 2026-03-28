cd /d "%~dp0"

set MODEL=Qwen/Qwen2.5-14B-Instruct
set DATASET=smartbugs-curated/dataset_cleaned_light
set PYTHON=.venv\Scripts\python.exe

%PYTHON% -m src.experiment --model %MODEL% --dataset %DATASET% --prompt ORIGINAL_ZS_COT --role --runs 5
if errorlevel 1 pause

%PYTHON% -m src.experiment --model %MODEL% --dataset %DATASET% --prompt ZS_COT --role --runs 5
if errorlevel 1 pause

pause
cd /d "%~dp0"

set MODEL=microsoft/Phi-3.5-mini-instruct
set DATASET=smartbugs-curated/dataset_cleaned_light
set PYTHON=.venv\Scripts\python.exe
set RUNS=1

%PYTHON% -m src.experiment --model %MODEL% --dataset %DATASET% --prompt ZS_COT --role --parser_mode sa --runs %RUNS%
if errorlevel 1 pause

%PYTHON% -m src.experiment --model %MODEL% --dataset %DATASET% --prompt ZS_COT --role --parser_mode deterministic --runs %RUNS%
if errorlevel 1 pause

pause
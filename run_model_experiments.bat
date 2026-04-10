cd /d "%~dp0"

set PYTHON=.venv\Scripts\python.exe
set DATASET=smartbugs-curated/dataset_cleaned_light

if "%~1"=="" (
    echo Usage:
    echo run_experiments.bat model_name
    pause
    exit /b 1
)

set MODEL=%~1

for %%P in (ZS_COT ZS FS) do (
    %PYTHON% -m src.experiment --model %MODEL% --dataset %DATASET% --prompt %%P --role --parser_mode sa --runs 5
    if errorlevel 1 pause
)

for %%P in (ZS_COT ZS FS) do (
    %PYTHON% -m src.experiment --model %MODEL% --dataset %DATASET% --prompt %%P --parser_mode sa --runs 5
    if errorlevel 1 pause
)

pause
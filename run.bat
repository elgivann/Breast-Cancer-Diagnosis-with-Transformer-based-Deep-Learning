@echo off

python -m pip show tqdm >nul 2>&1 || (
    echo Installing tqdm for progress bars...
    python -m pip install tqdm
)

for %%E in (50 100 200) do (
    echo ===========================================
    echo Training on BUS CoT with %%E epochs...
    echo ===========================================
    python -u main_reformed.py dataBUS-CoT.csv -t -e %%E -m models/model_%%E.pth

    echo.
    echo ===========================================
    echo Testing on BUS BRA with %%E epochs...
    echo ===========================================
    python -u main_reformed.py dataBUSBRA.csv -m models/model_%%E.pth -o out_%%E.csv --tta 10
    echo.
)

echo All runs completed.
pause

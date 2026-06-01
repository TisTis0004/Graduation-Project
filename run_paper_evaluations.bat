@echo off
echo =========================================================
echo PAPER EVALUATIONS: Data Caching and Evaluation Pipeline
echo =========================================================

echo.
echo [1] Generating 21-Channel Dataset (Binary)...
python data/cache_window_unipolar_21.py
if %errorlevel% neq 0 (
    echo Error during 21-channel data caching. Exiting.
    exit /b %errorlevel%
)

echo.
echo [2] Evaluating 21-Channel Model (Binary)...
python eval_single_model.py --ckpt "checkpoints/eegnet_10sec_full_next60.pt" --manifest "cache_windows_unipolar_21_eval/manifest.jsonl" --n_chans 21 --n_classes 2
if %errorlevel% neq 0 (
    echo Error during 21-channel evaluation. Exiting.
    exit /b %errorlevel%
)

echo.
echo =========================================================
echo ATTENTION REQUIRED FOR 41-CHANNEL MODEL:
echo Before running the 41-channel pipeline, you MUST edit:
echo data\cache_window_unipolar_41.py
echo And fill in the CANONICAL_41 list with your exact 41 channels!
echo =========================================================
echo.
pause

echo [3] Generating 41-Channel Dataset (9-Class Multiclass)...
python data/cache_window_unipolar_41.py
if %errorlevel% neq 0 (
    echo Error during 41-channel data caching. Exiting.
    exit /b %errorlevel%
)

echo.
echo [4] Evaluating 41-Channel Model (9-Class Multiclass)...
python eval_single_model.py --ckpt "checkpoints/best_model_checkpoint.pt" --manifest "cache_windows_unipolar_41_multiclass_eval/manifest.jsonl" --n_chans 41 --n_classes 9
if %errorlevel% neq 0 (
    echo Error during 41-channel evaluation. Exiting.
    exit /b %errorlevel%
)

echo.
echo =========================================================
echo SUCCESS!
echo All evaluations finished. Please check the 'assets' folder
echo for the confusion matrix and AUC plots!
echo =========================================================

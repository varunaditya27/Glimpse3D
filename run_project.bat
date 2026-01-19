@echo off
echo ===================================================
echo   Glimpse3D: Launcher (Debug Mode)
echo ===================================================
echo Current Directory: %CD%

echo.
echo [1/4] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH! 
    echo Please make sure you have Python installed and added to PATH.
    echo If using pyenv, run this from a terminal where pyenv is active.
    pause
    exit /b
)

echo.
echo [2/4] Verifying Environment...
python scripts/verify_full_stack.py
if %errorlevel% neq 0 (
    echo [ERROR] Verification failed.
    echo Please fix the errors above.
    pause
    exit /b
)

echo.
echo [3/4] Checking Weights...
if not exist "ai_modules/sync_dreamer/checkpoints/syncdreamer-pretrain.ckpt" (
    echo [WARNING] SyncDreamer weights missing!
    echo Please run: python scripts/download_weights.py
    pause
)

echo.
echo [4/4] Starting Services...
echo Launching Backend Server...
start "Glimpse3D Backend" cmd /k "echo STARTING BACKEND... && python -m backend.app.main"

echo Launching Frontend...
if exist "frontend" (
    start "Glimpse3D Frontend" cmd /k "echo STARTING FRONTEND... && cd frontend && npm run dev"
) else (
    echo [ERROR] Frontend folder not found!
)

echo.
echo ===================================================
echo   Startup Initiated.
echo   - Backend logs will appear in the Backend window.
echo   - Frontend logs will appear in the Frontend window.
echo   DO NOT CLOSE THIS WINDOW yet.
echo ===================================================
pause

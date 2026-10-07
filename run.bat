@echo off
REM One-click launcher for Nova (Windows). Creates the venv on first run.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    python -m venv .venv || goto :error
    echo [setup] Installing requirements...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

if not exist ".env" copy ".env.example" ".env" >nul

if /I "%1"=="api" (
    echo [run] Starting Nova API at http://localhost:8000/docs ...
    ".venv\Scripts\python.exe" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
    goto :eof
)

echo [run] Starting Nova at http://localhost:8501 ...
".venv\Scripts\python.exe" -m streamlit run streamlit_app_pro.py
goto :eof

:error
echo.
echo Setup failed. See the messages above.
pause

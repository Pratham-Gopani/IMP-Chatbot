@echo off
REM Launches MO-Insight locally on Windows: no cloud calls, everything on localhost.
cd /d "%~dp0"

if not exist ".env" (
    echo No .env found - copying .env.example -^> .env
    copy .env.example .env >nul
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt

echo.
echo Reminder: make sure Ollama is running locally (ollama serve) and that
echo the model set in .env (OLLAMA_MODEL) has been pulled (ollama pull ^<model^>).
echo.

echo Starting MO-Insight on http://0.0.0.0:8000 ...
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

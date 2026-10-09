#!/usr/bin/env bash
# Launches MO-Insight locally: no cloud calls, everything on localhost.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f ".env" ]; then
  echo "No .env found — copying .env.example -> .env (edit it if needed)"
  cp .env.example .env
fi

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

export $(grep -v '^#' .env | xargs -d '\n' 2>/dev/null || grep -v '^#' .env | xargs)

echo ""
echo "Checking for a local Ollama runtime at ${OLLAMA_BASE_URL:-http://localhost:11434} ..."
if ! curl -sf "${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" >/dev/null; then
  echo "WARNING: Ollama does not appear to be running."
  echo "Start it first (e.g. 'ollama serve' and 'ollama pull ${OLLAMA_MODEL:-qwen3.5:4b}')."
fi

echo ""
echo "Starting MO-Insight on http://${HOST:-0.0.0.0}:${PORT:-8000} ..."
cd backend
uvicorn main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" --reload

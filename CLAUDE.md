# CLAUDE.md - Project Guidelines for MO-Insight Universal Local Copilot

## Project Overview
MO-Insight is an enterprise-grade, 100% privacy-compliant Hybrid-RAG financial copilot.
It combines dynamic local relational CSV data, unstructured vector-indexed product PDFs,
and secure external market APIs under a unified agentic router running on local LLMs via Ollama.

## Tech Stack & Core Dependencies
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Structured Engine**: DuckDB with `watchdog` for dynamic hot-reloading of local CSV paths
- **Vector Engine**: ChromaDB (1.x) with local sentence-transformers embedding, also watched live via `watchdog`
- **LLM Runner**: Ollama (`langchain-ollama`) configured via environment variables
- **Frontend**: Single-page institutional UI (vanilla HTML/CSS/JS, Deep Navy `#002D62` palette)

## Repository Structure
```
mo-insight/
├── data/
│   ├── csvs/         # Live relational source files (brokerage.csv, imp_view.csv, current_portfolio.csv)
│   └── pdfs/         # Product brochures and model portfolio leaflets
├── backend/
│   ├── main.py       # FastAPI application, lifespan startup, routes
│   ├── database.py   # DuckDB auto-loader + watchdog hot-reload
│   ├── rag.py        # ChromaDB vector store + PDF chunking/embedding
│   ├── router.py      # Agentic intent router + citation synthesis
│   ├── external.py    # External market-data REST connector
│   └── llm.py          # Local LLM runtime wrapper (langchain-ollama)
├── static/
│   └── index.html    # Minimalist corporate chat UI
├── .env.example      # Template for local configuration
├── requirements.txt  # Pinned dependency manifest
├── run.sh / run.bat  # One-shot local launch scripts
├── Dockerfile / docker-compose.yml
└── CLAUDE.md
```

## Setup & Execution Commands
- **Environment Setup**: Copy `.env.example` to `.env` and configure local paths.
- **Install Dependencies**: `pip install -r requirements.txt`
- **Run Locally (Linux/Mac)**: `./run.sh`
- **Run Locally (Windows)**: `run.bat`
- **Run via Docker**: `docker compose up --build` (also launches an Ollama container)
- **Pull an LLM first**: `ollama pull qwen3.5:4b` (or set `OLLAMA_MODEL` to whatever you've pulled)
- **Changing models later**: the model name is hardcoded in exactly one place, the `OLLAMA_MODEL`
  default in `backend/llm.py` (overridable via `.env`, which normally means you never touch the
  Python file at all). Every other module (`router.py`, `main.py`, ...) only calls `get_llm()` and
  is otherwise model-agnostic. Also check `OLLAMA_NUM_CTX` if the new model's usable context
  differs meaningfully from the current 4096 default.
- **Slow / VRAM-constrained hardware**: `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE`, and `OLLAMA_NUM_GPU`
  are all set in `backend/llm.py` (overridable via `.env` — see the comments above each in
  `.env.example`). Run `ollama ps` while a request is in flight to check the CPU/GPU split for
  whatever's currently loaded; anything less than 100% GPU means it's spilling into much slower
  CPU inference. If a 9B model doesn't comfortably fit a given GPU's VRAM even after lowering
  `OLLAMA_NUM_CTX`, switching `OLLAMA_MODEL` to a smaller pull (e.g. a 3B–7B class model) is the
  more reliable fix than any code-side change — no amount of app-level concurrency makes a single
  local GPU generate tokens faster for one request.

## Development Rules & Coding Standards
1. **Zero Hardcoding**: Never hardcode client counts, business-associate hierarchies, brokerage
   sums, or product attributes. All metrics must be computed dynamically via DuckDB
   (`database.py`) or retrieved via ChromaDB vector search (`rag.py`).
2. **Absolute Data Privacy**: Ensure zero telemetry leaks. All LLM inference, vector embeddings,
   and database interactions must execute locally or via secured local endpoints (`localhost`).
   Do not add any third-party analytics or logging integrations.
3. **Mandatory Citations**: Every response containing factual or numerical data must include
   explicit source attribution matching one of:
   - `[Source: CSV::{filename} | Table: {table_name}]`
   - `[Source: PDF::{filename}, Page {number}]`
   - `[Source: API::{service_name} | Timestamp: {utc_time}]`
   This is enforced centrally in `router.py::HybridRouter.answer` — do not bypass it when adding
   new engines or endpoints.
4. **Read-only structured queries**: `database.py::StructuredDataEngine.run_sql` rejects any SQL
   containing write/DDL keywords. Do not weaken this guard.
5. **New CSVs/PDFs just work**: Dropping a new file into `data/csvs/` or `data/pdfs/`, or
   removing one, requires no restart — a `watchdog` observer for each directory
   (`database.py::start_watcher`, `rag.py::start_watcher`, both started from `main.py`'s
   lifespan) hot-reloads/re-embeds on create/modify and drops the corresponding table/chunks
   on delete. Keep new features consistent with this "drop-in" philosophy.
6. **Paths are resolved against the project root, not the process cwd**: `CSV_DATA_DIR`,
   `PDF_DATA_DIR`, `DUCKDB_PATH`, and `CHROMA_PATH` all resolve relative values against
   `Path(__file__).resolve().parent.parent` in `database.py`/`rag.py`, not the shell's current
   directory — this is what lets `run.sh`/`run.bat` `cd backend` before launching uvicorn
   without silently pointing at an empty `backend/data/`. Don't reintroduce a bare
   `Path(os.getenv(...))` default without that resolution.

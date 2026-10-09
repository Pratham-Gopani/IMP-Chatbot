# MO-Insight — Universal Local Hybrid-RAG Financial Copilot

A 100% local, privacy-compliant AI copilot for business associates and wealth managers.
It answers questions across three kinds of data at once:

1. **Structured** — client portfolios, brokerage/revenue, business-associate hierarchies
   (from CSVs, auto-loaded into DuckDB, hot-reloaded on file changes)
2. **Unstructured** — Model Portfolio product leaflets (PDFs, chunked and embedded locally
   into ChromaDB with sentence-transformers)
3. **External** — live market quotes (via a configurable REST connector)

An agentic router classifies each question, queries the right engine(s), and asks a local
LLM (via Ollama) to synthesize a single answer — always with explicit citations back to the
exact CSV table, PDF page, or API call that grounded it.

Nothing leaves your machine: no cloud telemetry, no third-party model training on your data.

## Quickstart

### 1. Install Ollama and pull a model
```bash
# https://ollama.com
ollama serve &
ollama pull qwen3.5:4b
```

**Changing the model later**: pull the new one, then set `OLLAMA_MODEL` in your `.env` to its
exact Ollama tag (e.g. `OLLAMA_MODEL=llama3.2:8b`) and restart. That's the only change needed —
see the comment above `OLLAMA_MODEL` in `backend/llm.py` for exactly why nothing else in the
codebase references a model name.

### 2. Run MO-Insight

**Linux / macOS**
```bash
./run.sh
```

**Windows**
```bat
run.bat
```

**Docker (bundles Ollama too)**
```bash
docker compose up --build
```

Then open **http://localhost:8000**.

## Sample data included
This project ships with de-identified sample data so you can try it immediately:

- `data/csvs/brokerage.csv` — trade-level brokerage/revenue records
- `data/csvs/imp_view.csv` — Intelligent Model Portfolio subscription records
- `data/csvs/current_portfolio.csv` — current client portfolio valuations
- `data/pdfs/*.pdf` — 14 Model Portfolio brochures (MO Signature, Technical Focus,
  RS Alpha Bluechip, MO 20 Quant Momentum, Zodiac, Signia, TM Prime, AN AlphaNext,
  AN Alpha Pulse, ASPIRE, APEX, VS Bluechip Edge, ASCEND, and the combined-alpha leaflet)

Drop your own CSVs into `data/csvs/` or PDFs into `data/pdfs/` at any time — they're picked
up automatically, no restart or code change required.

## Example questions to try
- "Which business associate earned the most brokerage this year?"
- "What is the minimum investment and benchmark for the MO Signia Model Portfolio?"
- "Show me the current profit value for client 1000101's portfolio."
- "Compare the risk profile of ASCEND vs APEX."
- "What's the live price of HDFCBANK?" *(requires internet access to NSE from your machine)*

## Architecture
```
                     ┌──────────────────────┐
   User question ──▶ │   Agentic Router      │
                     │  (intent classifier)  │
                     └──────┬───────┬───────┬┘
                 STRUCTURED │ UNSTRUCTURED│  EXTERNAL
                            ▼            ▼             ▼
                     ┌───────────┐ ┌────────────┐ ┌───────────┐
                     │  DuckDB    │ │  ChromaDB   │ │  NSE REST │
                     │ (CSVs)     │ │ (PDFs)      │ │  connector│
                     └─────┬─────┘ └──────┬──────┘ └─────┬─────┘
                           └──────────────┼───────────────┘
                                          ▼
                                 Local LLM (Ollama)
                                 synthesizes answer +
                                 mandatory citations
                                          ▼
                                  FastAPI JSON response
                                          ▼
                              Minimalist institutional
                                    chat UI
```

See `CLAUDE.md` for coding standards and repository conventions, and the PRD this project
was built from is included in this chat for full functional/non-functional requirements.

## Configuration
All configuration lives in `.env` (copy from `.env.example`):

| Variable | Purpose |
|---|---|
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Which local LLM runtime/model to use |
| `CSV_DATA_DIR`, `DUCKDB_PATH` | Structured engine paths |
| `PDF_DATA_DIR`, `CHROMA_PATH`, `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | RAG engine tuning |
| `NSE_API_BASE_URL`, `EXTERNAL_API_TIMEOUT` | External market-data connector |
| `HOST`, `PORT` | Server bind address |

## Notes & limitations
- The bundled NSE quote connector requires outbound internet access from wherever you run
  MO-Insight; it is not required for the structured/unstructured engines to work. NSE's own
  bot-detection routinely returns `403` for non-browser clients regardless — the connector
  surfaces this as a normal, citation-ready error rather than crashing, but treat live quotes
  as best-effort, not guaranteed.
- First run will download the `all-MiniLM-L6-v2` embedding model (~90MB) and ingest all
  sample PDFs — this can take a minute or two.
- `run.sh` / Docker both check that Ollama is reachable before starting, but will still boot
  the API even if it isn't — structured/unstructured retrieval and citations still work without
  it, you just get the raw retrieved evidence instead of an LLM-synthesized answer until Ollama
  is running.
- **Windows**: keep `chromadb` on a 1.x release (already pinned in `requirements.txt`). 0.5.x
  pins an exact `chroma-hnswlib` version with no prebuilt Windows wheel, which forces a
  from-source build requiring Microsoft's C++ Build Tools. 1.x ships its own compiled wheel and
  installs cleanly with plain `pip install -r requirements.txt` — no compiler needed.

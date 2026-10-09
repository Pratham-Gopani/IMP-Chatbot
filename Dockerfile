FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf/chromadb/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0 \
    PORT=8000 \
    CSV_DATA_DIR=/app/data/csvs \
    PDF_DATA_DIR=/app/data/pdfs \
    DUCKDB_PATH=/app/local_warehouse.db \
    CHROMA_PATH=/app/chroma_db

EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

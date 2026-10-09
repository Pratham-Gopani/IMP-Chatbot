"""
Unstructured Data Engine — ChromaDB + sentence-transformers
--------------------------------------------------------------
Parses every PDF in a local directory (default: ./data/pdfs), recursively
chunks the extracted text, embeds each chunk locally (no network calls),
and stores it in a persistent local ChromaDB collection. Retrieval returns
page-level citations so every answer can point back to `[Source: PDF::file, Page N]`.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Set

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("mo_insight.rag")

# See database.py for why defaults/relative overrides are resolved against the
# project root rather than the process's current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_value: str | None, default_relative: str) -> Path:
    raw = env_value if env_value else default_relative
    path = Path(raw)
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


PDF_DIR = _resolve(os.getenv("PDF_DATA_DIR"), "data/pdfs")
CHROMA_DIR = str(_resolve(os.getenv("CHROMA_PATH"), "chroma_db"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
COLLECTION_NAME = "mo_product_leaflets"


def _wait_for_stable_size(path: Path, checks: int = 3, interval: float = 0.25) -> None:
    """Same rationale as database.py's identical helper: a watcher event can
    fire mid-copy/mid-save, and reading a large PDF at that instant is more
    likely to fail outright (truncated PDF structure) than silently ingest a
    partial file — but waiting for the file to stop growing first means the
    retry loop in _ingest_pdf is only ever needed for genuine lock/IO
    contention, not for "the file simply wasn't finished yet"."""
    last_size = -1
    stable_count = 0
    for _ in range(checks * 8):
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size == last_size:
            stable_count += 1
            if stable_count >= checks:
                return
        else:
            stable_count = 0
            last_size = size
        time.sleep(interval)


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple recursive-ish fixed-window chunker with overlap."""
    chunks: List[str] = []
    start, n = 0, len(text)
    if n == 0:
        return chunks
    step = max(size - overlap, 1)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


class UnstructuredDataEngine:
    def __init__(self, pdf_dir: Path = PDF_DIR):
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        # anonymized_telemetry=False: the PRD's zero-leak/air-gapped requirement
        # means Chroma must never phone home, even anonymized usage pings.
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=self.embed_fn
        )
        self.ingested_files: Set[str] = self._already_ingested()
        self.ingest_all()

    # ------------------------------------------------------------------
    def _already_ingested(self) -> Set[str]:
        if self.collection.count() == 0:
            return set()
        got = self.collection.get(include=["metadatas"])
        return {m["source_file"] for m in got.get("metadatas", []) if m}

    def ingest_all(self) -> None:
        for pdf_path in sorted(self.pdf_dir.glob("*.pdf")):
            if pdf_path.name in self.ingested_files:
                continue
            self._ingest_pdf(pdf_path)

    def _ingest_pdf(self, pdf_path: Path, _retries: int = 5, _backoff: float = 0.3) -> None:
        # A watcher event can fire while a large PDF is still being copied
        # in — retry briefly rather than permanently skipping the file (see
        # the identical rationale in database.py::_load_csv).
        for attempt in range(1, _retries + 1):
            try:
                reader = PdfReader(str(pdf_path))
                ids, docs, metas = [], [], []
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    for i, chunk in enumerate(_chunk_text(text)):
                        ids.append(f"{pdf_path.name}::p{page_num}::c{i}")
                        docs.append(chunk)
                        metas.append({"source_file": pdf_path.name, "page": page_num})
                if docs:
                    self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
                    self.ingested_files.add(pdf_path.name)
                    logger.info("Ingested %s (%d chunks)", pdf_path.name, len(docs))
                else:
                    logger.warning("No extractable text found in %s", pdf_path.name)
                return
            except Exception as exc:  # noqa: BLE001
                if attempt < _retries:
                    time.sleep(_backoff)
                    continue
                logger.error("Failed to ingest %s after %d attempts: %s", pdf_path, attempt, exc)

    def reindex(self, filename: str) -> None:
        """Force re-ingestion of a single file (e.g. after it changed on disk)."""
        path = self.pdf_dir / filename
        if not path.exists():
            return
        _wait_for_stable_size(path)
        # Remove any existing chunks for this file first.
        existing = self.collection.get(where={"source_file": filename})
        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])
        self.ingested_files.discard(filename)
        self._ingest_pdf(path)

    def remove(self, filename: str) -> None:
        """Drop all chunks for a PDF that's been deleted from disk, so a
        withdrawn brochure can't still be cited/retrieved after removal."""
        existing = self.collection.get(where={"source_file": filename})
        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])
            logger.info("Removed %d chunks for deleted file %s", len(existing["ids"]), filename)
        self.ingested_files.discard(filename)

    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = 5) -> List[dict]:
        if self.collection.count() == 0:
            return []
        results = self.collection.query(query_texts=[question], n_results=top_k)
        hits: List[dict] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(
                {
                    "text": doc,
                    "source_file": meta["source_file"],
                    "page": meta["page"],
                    "distance": dist,
                }
            )
        return hits


class _PDFWatchHandler(FileSystemEventHandler):
    """Forwards created/modified/deleted .pdf events for live re-indexing —
    same drop-in-and-it-works behavior as the CSV watcher in database.py."""

    def __init__(self, engine: UnstructuredDataEngine):
        self.engine = engine

    def on_created(self, event):
        self._reindex(event)

    def on_modified(self, event):
        self._reindex(event)

    def on_deleted(self, event):
        if event.is_directory or not event.src_path.lower().endswith(".pdf"):
            return
        try:
            self.engine.remove(Path(event.src_path).name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Watcher failed to remove chunks for %s: %s", event.src_path, exc)

    def on_moved(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(".pdf"):
            self.engine.remove(Path(event.src_path).name)
        if event.dest_path.lower().endswith(".pdf"):
            self.engine.reindex(Path(event.dest_path).name)

    def _reindex(self, event):
        if event.is_directory or not event.src_path.lower().endswith(".pdf"):
            return
        try:
            self.engine.reindex(Path(event.src_path).name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Watcher failed to reindex %s: %s", event.src_path, exc)


def start_watcher(engine: UnstructuredDataEngine) -> Observer:
    observer = Observer()
    observer.schedule(_PDFWatchHandler(engine), str(engine.pdf_dir), recursive=False)
    observer.start()
    logger.info("Watching %s for live PDF changes", engine.pdf_dir)
    return observer

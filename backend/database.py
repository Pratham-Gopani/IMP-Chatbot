"""
Structured Data Engine — DuckDB
--------------------------------
Watches a local directory (default: ./data/csvs) and automatically loads
every CSV found there as a DuckDB table, keeping tables in sync with the
files on disk via a `watchdog` file-system observer. No hardcoded schemas,
table names, or column lists: everything is derived at load time from
whatever CSVs happen to be present, so new files "just work".

Table naming: the CSV's filename (without extension), lowercased and with
any non [a-z0-9_] character replaced by `_`.
  e.g. data/csvs/current_portfolio.csv -> table `current_portfolio`
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List

import duckdb
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("mo_insight.database")

# backend/database.py -> parent = backend/, parent.parent = project root. Resolving
# defaults (and any relative env override) against this instead of the process's
# current working directory means CSV ingestion works whether uvicorn is started
# from the project root, from backend/, or via Docker.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_value: str | None, default_relative: str) -> Path:
    raw = env_value if env_value else default_relative
    path = Path(raw)
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


CSV_DIR = _resolve(os.getenv("CSV_DATA_DIR"), "data/csvs")
DB_PATH = str(_resolve(os.getenv("DUCKDB_PATH"), "local_warehouse.db"))

# Any keyword below anywhere in a submitted query is treated as a write
# attempt and rejected — this engine is read-only from the LLM's point of view.
_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "attach", "detach", "copy", "pragma", "call", "export", "import",
)


def _table_name_from_path(path: Path) -> str:
    name = path.stem.lower()
    return re.sub(r"[^a-z0-9_]", "_", name)


def _wait_for_stable_size(path: Path, checks: int = 3, interval: float = 0.25) -> None:
    """A save from Excel, an export script, or a copy over a network drive
    can trigger a filesystem event WHILE the file is still being written.
    Reading it at that instant doesn't raise anything — read_csv_auto just
    happily loads however many rows had been flushed so far, and since
    nothing failed, the retry-on-exception loop in _load_csv never kicks in
    and no later event necessarily arrives to correct it. That's how a
    100-row CSV update can silently end up loaded as 50 rows: not a query
    bug, a load-time race. This waits until the file's size stops changing
    across a few consecutive checks before handing it to _load_csv, so a
    half-written file is never mistaken for the finished one. Bounded by a
    hard iteration cap so a file that's genuinely stuck (e.g. permanently
    open/locked elsewhere) can't hang the watcher thread forever — normal
    retry-on-exception handling in _load_csv still applies after this.
    """
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


class StructuredDataEngine:
    """Thread-safe wrapper around a DuckDB connection that auto-loads CSVs."""

    def __init__(self, csv_dir: Path = CSV_DIR, db_path: str = DB_PATH):
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.con = duckdb.connect(db_path)
        self.loaded_files: Dict[str, float] = {}  # path -> mtime at last load
        self.reload_all()

    # ------------------------------------------------------------------
    # Loading / hot-reload
    # ------------------------------------------------------------------
    def reload_all(self) -> None:
        with self._lock:
            for csv_path in sorted(self.csv_dir.glob("*.csv")):
                self._load_csv(csv_path)

    def _load_csv(self, csv_path: Path, _retries: int = 5, _backoff: float = 0.2) -> None:
        table = _table_name_from_path(csv_path)
        with self._lock:
            for attempt in range(1, _retries + 1):
                try:
                    self.con.execute(
                        f"CREATE OR REPLACE TABLE {table} AS "
                        "SELECT * FROM read_csv_auto(?, SAMPLE_SIZE=-1, IGNORE_ERRORS=TRUE)",
                        [str(csv_path)],
                    )
                    self.loaded_files[str(csv_path)] = csv_path.stat().st_mtime
                    # Logging the actual row count loaded (not just "success")
                    # is what makes a partial/truncated load visible at all —
                    # IGNORE_ERRORS=TRUE means a malformed row is silently
                    # skipped rather than raising, so without this a file
                    # with fewer rows loaded than it actually has would look
                    # identical to a clean, complete load in the logs.
                    row_count = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    logger.info(
                        "Loaded/refreshed table '%s' from %s (%d rows)", table, csv_path.name, row_count
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    # A file-watcher event can fire while a writer (Excel, an
                    # export script, ...) still holds the file open, especially
                    # on Windows. Retry briefly before giving up so a normal
                    # save doesn't permanently drop the table until some
                    # unrelated later event happens to touch this file again.
                    if attempt < _retries:
                        time.sleep(_backoff)
                        continue
                    logger.error(
                        "Failed to load %s into DuckDB after %d attempts: %s",
                        csv_path, attempt, exc,
                    )

    def handle_fs_event(self, path_str: str) -> None:
        path = Path(path_str)
        if path.suffix.lower() != ".csv":
            return
        if path.exists():
            _wait_for_stable_size(path)
            self._load_csv(path)
        else:
            # File was deleted/renamed away — drop the stale table so removed
            # data (e.g. a client's records) can't still be queried afterward.
            self._drop_table_for(path)

    def _drop_table_for(self, csv_path: Path) -> None:
        table = _table_name_from_path(csv_path)
        with self._lock:
            try:
                self.con.execute(f"DROP TABLE IF EXISTS {table}")
                self.loaded_files.pop(str(csv_path), None)
                logger.info("Dropped table '%s' — source file %s was removed", table, csv_path.name)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to drop table for removed file %s: %s", csv_path, exc)

    # ------------------------------------------------------------------
    # Schema introspection (fed to the LLM for text-to-SQL)
    # ------------------------------------------------------------------
    def schema_description(self) -> str:
        with self._lock:
            tables = self.con.execute("SHOW TABLES").fetchall()
            lines: List[str] = []
            for (table,) in tables:
                cols = self.con.execute(f"DESCRIBE {table}").fetchall()
                col_desc = ", ".join(f"{c[0]} ({c[1]})" for c in cols)
                lines.append(f"TABLE {table}: {col_desc}")
            return "\n".join(lines)

    def list_tables(self) -> List[str]:
        with self._lock:
            return [t[0] for t in self.con.execute("SHOW TABLES").fetchall()]

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def run_sql(self, sql: str, max_rows: int = 200):
        """Execute a read-only SELECT/WITH query and return a DataFrame."""
        cleaned = sql.strip().rstrip(";")
        lowered = f" {cleaned.lower()} "
        if not (cleaned.lower().startswith("select") or cleaned.lower().startswith("with")):
            raise ValueError("Only SELECT / WITH read-only queries are permitted.")
        if any(f" {kw} " in lowered for kw in _FORBIDDEN_KEYWORDS):
            raise ValueError("Query contains a disallowed keyword (write operations are blocked).")
        with self._lock:
            df = self.con.execute(cleaned).fetchdf()
        return df.head(max_rows)

    def run_sql_params(self, sql: str, params: list, max_rows: int = 200):
        """Like run_sql, but for backend-authored queries with a value that
        needs safe parameterization (e.g. a BA code in a dashboard endpoint)
        instead of raw string interpolation. Still enforces the same
        read-only guard — this is about injection-safety, not a way around it."""
        cleaned = sql.strip().rstrip(";")
        lowered = f" {cleaned.lower()} "
        if not (cleaned.lower().startswith("select") or cleaned.lower().startswith("with")):
            raise ValueError("Only SELECT / WITH read-only queries are permitted.")
        if any(f" {kw} " in lowered for kw in _FORBIDDEN_KEYWORDS):
            raise ValueError("Query contains a disallowed keyword (write operations are blocked).")
        with self._lock:
            df = self.con.execute(cleaned, params).fetchdf()
        return df.head(max_rows)


class _CSVWatchHandler(FileSystemEventHandler):
    """Forwards created/modified .csv events to the engine for hot-reload."""

    def __init__(self, engine: StructuredDataEngine):
        self.engine = engine

    def on_created(self, event):
        self._dispatch(event)

    def on_modified(self, event):
        self._dispatch(event)

    def on_deleted(self, event):
        self._dispatch(event)

    def on_moved(self, event):
        # A rename/move away is a deletion of the old path and (if still a
        # .csv) a creation at the new one.
        if event.is_directory:
            return
        try:
            self.engine.handle_fs_event(event.src_path)
            self.engine.handle_fs_event(event.dest_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Watcher failed to handle move %s -> %s: %s", event.src_path, event.dest_path, exc)

    def _dispatch(self, event):
        if event.is_directory:
            return
        try:
            self.engine.handle_fs_event(event.src_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Watcher failed to reload %s: %s", event.src_path, exc)


def start_watcher(engine: StructuredDataEngine) -> Observer:
    observer = Observer()
    observer.schedule(_CSVWatchHandler(engine), str(engine.csv_dir), recursive=False)
    observer.start()
    logger.info("Watching %s for live CSV changes", engine.csv_dir)
    return observer

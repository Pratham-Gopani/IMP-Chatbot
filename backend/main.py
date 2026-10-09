"""
MO-Insight — Universal Local Hybrid-RAG Financial Copilot
Secure Air-Gapped Edition

FastAPI entrypoint. Everything — LLM inference, vector search, relational
queries — runs on localhost. No data ever leaves this machine.
"""
import asyncio
import json
import logging
import queue as queue_module
import secrets
import threading
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# backend/main.py -> parent = backend/, parent.parent = project root. Load .env
# from there (not the process cwd) so `.env` works regardless of whether uvicorn
# is launched from the project root or from inside backend/, and do it BEFORE
# importing the engine modules below, since they read os.environ at import time.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests

import auth
import chat_history
from database import StructuredDataEngine
from database import start_watcher as start_csv_watcher
from llm import OLLAMA_BASE_URL, OLLAMA_MODEL
from rag import UnstructuredDataEngine
from rag import start_watcher as start_pdf_watcher
from router import MAX_HISTORY_TURNS, HybridRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("mo_insight.main")

ENGINES: dict = {}
STATIC_DIR = BASE_DIR / "static"

# In-memory, per-conversation (question, answer) history — never written to
# disk, consistent with the app's "transient context" privacy stance. Keyed
# by a client-generated conversation_id; the frontend mints a new one on
# "New conversation", which is what actually resets the bot's memory (an
# unrecognized/absent id just means "no history yet", not an error).
# Bounded on two axes so a long-running server can't grow unbounded: each
# conversation keeps only its last MAX_HISTORY_TURNS turns, and only
# MAX_CONVERSATIONS distinct conversations are retained (oldest evicted first).
MAX_CONVERSATIONS = 200
_conversations_lock = threading.Lock()
CONVERSATIONS: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()


def _get_history(conversation_id: Optional[str]) -> list[tuple[str, str]]:
    if not conversation_id:
        return []
    with _conversations_lock:
        return list(CONVERSATIONS.get(conversation_id, []))


def _append_turn(conversation_id: Optional[str], question: str, answer: str) -> None:
    if not conversation_id:
        return
    with _conversations_lock:
        history = CONVERSATIONS.setdefault(conversation_id, [])
        history.append((question, answer))
        del history[:-MAX_HISTORY_TURNS]
        CONVERSATIONS.move_to_end(conversation_id)
        while len(CONVERSATIONS) > MAX_CONVERSATIONS:
            CONVERSATIONS.popitem(last=False)


# Server-side session store — token -> profile ({username, role, ba_code, name}).
# This is the actual RBAC boundary: the client never gets to *tell* the server
# who it is on /api/chat or /api/dashboard (the old ba_code request field was
# spoofable — any client could just claim to be a different BA). Instead the
# client authenticates once via /api/login, gets an opaque token back, and
# every subsequent request's identity is looked up here, server-side, from
# that token. In-memory only and wiped on restart, consistent with this being
# a local single-machine demo, not a multi-instance production deployment.
_sessions_lock = threading.Lock()
SESSIONS: dict[str, dict] = {}


def _resolve_session(x_session_token: Optional[str]) -> dict:
    if not x_session_token:
        raise HTTPException(status_code=401, detail="Not logged in.")
    with _sessions_lock:
        profile = SESSIONS.get(x_session_token)
    if not profile:
        raise HTTPException(status_code=401, detail="Session expired or invalid — please log in again.")
    return profile


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Booting MO-Insight — loading structured + unstructured engines ...")
    structured = StructuredDataEngine()
    unstructured = UnstructuredDataEngine()

    ENGINES["structured"] = structured
    ENGINES["unstructured"] = unstructured
    ENGINES["router"] = HybridRouter(structured, unstructured)
    ENGINES["csv_observer"] = start_csv_watcher(structured)
    ENGINES["pdf_observer"] = start_pdf_watcher(unstructured)

    logger.info("MO-Insight ready. Structured tables: %s", structured.list_tables())
    yield

    for key in ("csv_observer", "pdf_observer"):
        observer = ENGINES.get(key)
        if observer:
            observer.stop()
            observer.join()
    logger.info("MO-Insight shut down cleanly.")


app = FastAPI(title="MO-Insight Copilot", version="1.0.0", lifespan=lifespan)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    role: str
    ba_code: Optional[str] = None
    name: str


@app.post("/api/login", response_model=LoginResponse)
async def login(req: LoginRequest) -> LoginResponse:
    profile = auth.authenticate(req.username, req.password)
    if not profile:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = secrets.token_urlsafe(32)
    with _sessions_lock:
        SESSIONS[token] = profile
    return LoginResponse(token=token, **profile)


@app.post("/api/logout")
async def logout(x_session_token: Optional[str] = Header(default=None)):
    if x_session_token:
        with _sessions_lock:
            SESSIONS.pop(x_session_token, None)
    return {"ok": True}


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


# A sentinel distinct from any real queue payload, so the consumer can tell
# "the worker thread is completely done" apart from a legitimate falsy event.
_STREAM_DONE = object()


@app.post("/api/chat")
async def chat(req: ChatRequest, x_session_token: Optional[str] = Header(default=None)):
    """Streams the answer as newline-delimited JSON events instead of
    returning one blocking response — {"type": "chunk", "text": "..."} as
    the LLM produces each piece, then one {"type": "done", "intent": ...,
    "citations": [...], "answer": full_text} once generation finishes. This
    is what lets the UI render the answer as it's written instead of showing
    a spinner for the whole (often 10-30s) generation and then dumping the
    entire answer on screen at once.
    """
    profile = _resolve_session(x_session_token)
    if "router" not in ENGINES:
        raise HTTPException(status_code=503, detail="Engines still starting up, try again shortly.")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    history = _get_history(req.conversation_id)
    router = ENGINES["router"]

    # HybridRouter.answer_stream() is a fully synchronous generator (blocking
    # HTTP calls to Ollama, DuckDB, ChromaDB) — iterating it directly on the
    # event loop would stall every other concurrent user for the whole
    # generation, so it runs on a background thread that pushes each event
    # into a thread-safe queue, which the async generator below drains.
    event_queue: "queue_module.Queue" = queue_module.Queue()

    def _run_in_thread():
        try:
            for event in router.answer_stream(req.message, history, profile["ba_code"]):
                event_queue.put(event)
        except Exception as exc:  # noqa: BLE001
            event_queue.put({"type": "error", "detail": str(exc)})
        finally:
            event_queue.put(_STREAM_DONE)

    threading.Thread(target=_run_in_thread, daemon=True).start()

    async def event_stream():
        loop = asyncio.get_event_loop()
        while True:
            event = await loop.run_in_executor(None, event_queue.get)
            if event is _STREAM_DONE:
                break
            if event.get("type") == "done":
                _append_turn(req.conversation_id, req.message, event["answer"])
                chat_history.append_turn(
                    profile["username"],
                    req.conversation_id or "",
                    req.message,
                    event["answer"],
                    event["intent"],
                    event["citations"],
                )
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/api/history")
async def history_endpoint(
    target: Optional[str] = None,
    conversation_id: Optional[str] = None,
    x_session_token: Optional[str] = Header(default=None),
):
    """A BA only ever sees their own chat history. An admin may pass `target`
    (a username) to view any user's history instead of their own.

    Two shapes, depending on `conversation_id`:
      - omitted -> the chat LIST: one summary (title, timestamps, turn
        count) per past conversation, most recent first — see
        chat_history.list_conversations.
      - provided -> the full THREAD for that one conversation, every turn
        in order — see chat_history.load_conversation.
    """
    profile = _resolve_session(x_session_token)
    is_admin = profile["role"] == "admin"
    # Non-admins can't view anyone else's history — same fail-closed pattern
    # as the rest of this app's RBAC: an ignored/overridden `target` rather
    # than an error, matching this endpoint's pre-existing behavior.
    username = target if (is_admin and target) else profile["username"]

    if conversation_id:
        return {
            "username": username,
            "conversation_id": conversation_id,
            "turns": chat_history.load_conversation(username, conversation_id),
        }

    result = {"username": username, "conversations": chat_history.list_conversations(username)}
    if is_admin and not target:
        result["all_users"] = chat_history.list_users_with_history()
    return result


@app.get("/api/bas")
async def list_bas():
    """Real BA directory for the identity picker — not privileged data (BA
    codes/names are internal-directory information, not client PII), and
    deliberately NOT filtered by who's asking: everyone needs to see the
    full list to pick who *they* are in the first place."""
    structured = ENGINES.get("structured")
    if not structured or "current_portfolio" not in structured.list_tables():
        return {"bas": []}
    df = structured.run_sql(
        "SELECT DISTINCT BA1200code AS code, BAName AS name FROM current_portfolio ORDER BY BA1200code"
    )
    return {"bas": df.to_dict(orient="records")}


@app.get("/api/dashboard")
async def dashboard(x_session_token: Optional[str] = Header(default=None)):
    """Real, RBAC-scoped KPIs for the Dashboard view. Identity comes from the
    server-side session (never a client-supplied query param — that was
    spoofable): admin sees firm-wide numbers and a cross-BA leaderboard, a BA
    is hard-scoped to their own clients/brokerage only, mirroring the same
    access rule enforced for chat in router.py."""
    profile = _resolve_session(x_session_token)
    ba_code = profile["ba_code"]
    structured = ENGINES.get("structured")
    if not structured:
        raise HTTPException(status_code=503, detail="Engines still starting up, try again shortly.")

    if ba_code:
        clients_df = structured.run_sql_params(
            "SELECT COUNT(DISTINCT ClientCode) AS n FROM current_portfolio WHERE BA1200code = ?",
            [ba_code],
        )
        aum_df = structured.run_sql_params(
            "SELECT COALESCE(SUM(CurrentValue), 0) AS total FROM current_portfolio WHERE BA1200code = ?",
            [ba_code],
        )
        brokerage_df = structured.run_sql_params(
            "SELECT COALESCE(SUM(brokerageamount), 0) AS total FROM brokerage WHERE referalbacode = ?",
            [ba_code],
        )
        top_clients_df = structured.run_sql_params(
            "SELECT ClientCode AS code, ClientName AS name, CurrentValue AS value "
            "FROM current_portfolio WHERE BA1200code = ? ORDER BY CurrentValue DESC LIMIT 5",
            [ba_code],
        )
        return {
            "scope": "ba",
            "ba_code": ba_code,
            "total_clients": int(clients_df["n"].iloc[0]),
            "total_aum": float(aum_df["total"].iloc[0]),
            "total_brokerage": float(brokerage_df["total"].iloc[0]),
            "top_clients": top_clients_df.to_dict(orient="records"),
            "leaderboard": [],
        }

    clients_df = structured.run_sql("SELECT COUNT(DISTINCT ClientCode) AS n FROM current_portfolio")
    aum_df = structured.run_sql("SELECT COALESCE(SUM(CurrentValue), 0) AS total FROM current_portfolio")
    brokerage_df = structured.run_sql("SELECT COALESCE(SUM(brokerageamount), 0) AS total FROM brokerage")
    top_clients_df = structured.run_sql(
        "SELECT ClientCode AS code, ClientName AS name, CurrentValue AS value "
        "FROM current_portfolio ORDER BY CurrentValue DESC LIMIT 5"
    )
    leaderboard_df = structured.run_sql(
        "SELECT referalbacode AS code, SUM(brokerageamount) AS total FROM brokerage "
        "GROUP BY referalbacode ORDER BY total DESC LIMIT 5"
    )
    return {
        "scope": "admin",
        "ba_code": None,
        "total_clients": int(clients_df["n"].iloc[0]),
        "total_aum": float(aum_df["total"].iloc[0]),
        "total_brokerage": float(brokerage_df["total"].iloc[0]),
        "top_clients": top_clients_df.to_dict(orient="records"),
        "leaderboard": leaderboard_df.to_dict(orient="records"),
    }


@app.get("/api/documents")
async def list_documents():
    """Real filenames from data/pdfs — Model Portfolio leaflets are general
    product literature, not client data, so this is the same for every role."""
    unstructured = ENGINES.get("unstructured")
    if not unstructured:
        return {"documents": []}
    files = sorted(p.name for p in unstructured.pdf_dir.glob("*.pdf"))
    return {"documents": files}


def _ollama_reachable() -> bool:
    """A quick, cheap ping to Ollama's own API (NOT a chat call) — just
    confirms the runtime is up and the configured model has actually been
    pulled, so a stale/wrong OLLAMA_MODEL or a not-yet-`ollama pull`-ed model
    shows up immediately in /api/health instead of only surfacing later as
    slow, degraded, or missing structured-data answers."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        resp.raise_for_status()
        pulled = {m.get("name") for m in resp.json().get("models", [])}
        # Ollama tags omit an explicit ":latest" suffix in some responses —
        # match either the exact configured tag or its bare name.
        base_name = OLLAMA_MODEL.split(":")[0]
        return any(OLLAMA_MODEL == name or name.startswith(base_name + ":") for name in pulled)
    except Exception:  # noqa: BLE001
        return False


@app.get("/api/health")
async def health():
    structured = ENGINES.get("structured")
    unstructured = ENGINES.get("unstructured")
    return {
        "status": "ok" if structured else "starting",
        "structured_tables": structured.list_tables() if structured else [],
        "pdf_chunks_indexed": unstructured.collection.count() if unstructured else 0,
        "ollama_model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model_pulled": _ollama_reachable(),
    }


# Static frontend -----------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/favicon.ico")
async def favicon():
    # index.html already declares a data-URI <link rel="icon">, which modern
    # browsers use directly — but some clients still probe /favicon.ico
    # unconditionally. Answer with 204 instead of a 404 so it doesn't spam
    # the server log on every page load.
    return Response(status_code=204)

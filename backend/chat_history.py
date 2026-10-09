"""
Durable, per-user chat history — appended to a local file on every turn.
--------------------------------------------------------------------------
This is independent of the short-term in-memory conversation buffer in
main.py (CONVERSATIONS), which only exists to give the LLM recent context
and is intentionally wiped whenever "New Chat" is clicked, or the server
restarts. This module is the permanent record: "what did this user ask, and
what did the bot answer, ever" — written as newline-delimited JSON (one
record per line) to disk, so it survives restarts, is human-readable, is
easy to grep/audit, and is safe to append to without loading/rewriting the
whole file each time.

Deliberately stored OUTSIDE data/csvs and data/pdfs — those two directories
are auto-loaded into DuckDB/ChromaDB and become queryable by the chat itself.
Chat transcripts (which can contain a BA's own client names/figures) must
never end up as a table the LLM — or a DIFFERENT BA — can query over.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(env_value: Optional[str], default_relative: str) -> Path:
    raw = env_value if env_value else default_relative
    path = Path(raw)
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


# Override via CHAT_HISTORY_DIR in .env if you want this somewhere specific
# (e.g. a literal path on the Desktop) — defaults to a project-relative
# folder, consistent with every other data path in this app.
HISTORY_DIR = _resolve(os.getenv("CHAT_HISTORY_DIR"), "chat_history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


def _safe_filename(username: str) -> str:
    # Usernames are our own account names ("admin", "BA12005", ...) today,
    # but this touches the filesystem, so never trust it blindly regardless.
    safe = "".join(c for c in username if c.isalnum() or c in ("-", "_")).strip()
    return (safe or "unknown") + ".jsonl"


def append_turn(
    username: str,
    conversation_id: str,
    question: str,
    answer: str,
    intent: str,
    citations: List[str],
) -> None:
    path = HISTORY_DIR / _safe_filename(username)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer,
        "intent": intent,
        "citations": citations,
    }
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_all_records(username: str) -> List[dict]:
    """Every turn ever recorded for this user, oldest first, across every
    conversation — the raw material that load_history/list_conversations/
    load_conversation below all group differently. Unlike the old
    load_history's tail-based reading, this always reads the whole file:
    grouping by conversation_id needs every record, not just the most
    recent N turns regardless of which chat they belonged to."""
    path = HISTORY_DIR / _safe_filename(username)
    if not path.exists():
        return []
    with _lock:
        text = path.read_text(encoding="utf-8")
    records = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def load_history(username: str, limit: int = 200) -> List[dict]:
    """Most recent `limit` turns, oldest first, across every conversation —
    a flat view. Kept for callers that just want "recent activity"; the
    chat-history UI itself uses list_conversations/load_conversation below
    instead, since a flat list stops being readable once someone has had
    more than a couple of separate conversations."""
    return _read_all_records(username)[-limit:]


_TITLE_MAX_LEN = 60


def _derive_title(first_question: str) -> str:
    """Names a chat the same way this app already names a "New Chat" in
    the sidebar before you've typed a title yourself — a short version of
    the first thing you asked, not a generic "Conversation #4". Purely
    local, deterministic string work: no LLM call, so naming a chat costs
    nothing and needs no network, in keeping with keeping requests-per-
    question down (see router.py's contextualize_and_classify)."""
    text = re.sub(r"\s+", " ", first_question or "").strip()
    if not text:
        return "New chat"
    if len(text) <= _TITLE_MAX_LEN:
        return text
    truncated = text[:_TITLE_MAX_LEN].rsplit(" ", 1)[0].rstrip(",.;:")
    return (truncated or text[:_TITLE_MAX_LEN]) + "…"


def list_conversations(username: str) -> List[dict]:
    """Every distinct conversation this user has had, most recently active
    first — one summary card per chat, each with a short auto-generated
    title, rather than one continuous flat feed of every question ever
    asked mixed together. Each conversation is identified by the
    conversation_id already recorded on every turn (the same ID the
    frontend mints per "New Chat" and reuses until the next one)."""
    conversations: "dict[str, dict]" = {}
    for rec in _read_all_records(username):
        cid = rec.get("conversation_id") or "unknown"
        conv = conversations.get(cid)
        if conv is None:
            conv = {
                "conversation_id": cid,
                "title": _derive_title(rec.get("question", "")),
                "started_at": rec.get("timestamp_utc"),
                "last_message_at": rec.get("timestamp_utc"),
                "first_question": rec.get("question", ""),
                "turn_count": 0,
            }
            conversations[cid] = conv
        conv["last_message_at"] = rec.get("timestamp_utc")
        conv["turn_count"] += 1
    return sorted(conversations.values(), key=lambda c: c["last_message_at"] or "", reverse=True)


def load_conversation(username: str, conversation_id: str) -> List[dict]:
    """Every turn belonging to ONE specific conversation, oldest first —
    the full-thread view a user gets after picking a chat from
    list_conversations above."""
    return [r for r in _read_all_records(username) if r.get("conversation_id") == conversation_id]


def list_users_with_history() -> List[str]:
    """Every username that has ever chatted — used for the admin's
    "view any user's history" picker."""
    return sorted(p.stem for p in HISTORY_DIR.glob("*.jsonl"))

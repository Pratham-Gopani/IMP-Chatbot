"""
Dummy local login — NOT production-grade security.
--------------------------------------------------
This is a demo credential check for a fully local, air-gapped internal tool
with no real identity provider to integrate with (no LDAP/SSO/employee
directory to check against). Passwords are stored as SHA-256 hashes in
users.json rather than plaintext, as a baseline precaution — but there is no
rate limiting, no account lockout, no session expiry, no HTTPS enforcement
(moot on localhost anyway), and no protection against someone reading
users.json directly off disk. Treat this as "stops the app from casually
showing the wrong BA's data", not as a real defense against a determined
attacker with access to this machine. Replace with real SSO before this ever
leaves a single trusted local demo.

users.json lives at the project root (NOT under data/csvs or data/pdfs) —
those two directories are auto-loaded into DuckDB/ChromaDB and become
queryable by the chat itself. Credentials must never end up somewhere the
LLM could be asked to recite them.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
USERS_FILE = Path(os.getenv("USERS_FILE") or (_PROJECT_ROOT / "users.json"))


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Strip the human-readable reference/comment keys — they're not accounts.
    return {k: v for k, v in data.items() if not k.startswith("_")}


def authenticate(username: str, password: str) -> Optional[dict]:
    """Returns {username, role, ba_code, name} on success, None on failure.
    role is "admin" or "ba"; ba_code is None for admin."""
    if not username or not password:
        return None
    users = _load_users()
    user = users.get(username.strip())
    if not user:
        return None
    if _hash(password) != user.get("password_hash"):
        return None
    return {
        "username": username.strip(),
        "role": user["role"],
        "ba_code": user.get("ba_code"),
        "name": user["name"],
    }

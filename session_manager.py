"""
session_manager.py
------------------
Manages multi-session isolation, persistent session registry, and session-specific
Chroma vector stores — ensuring full context isolation between chats.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from logger_config import logger

BASE_CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
SESSIONS_DIR = os.path.join(BASE_CHROMA_DIR, "sessions")
REGISTRY_FILE = os.path.join(SESSIONS_DIR, "sessions_registry.json")


def _ensure_dirs():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def get_session_db_dir(session_id: str) -> str:
    """Returns the isolated disk directory for a session's ChromaDB."""
    _ensure_dirs()
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def load_registry() -> Dict[str, Any]:
    """Loads the sessions registry from disk."""
    _ensure_dirs()
    if not os.path.exists(REGISTRY_FILE):
        return {"active_session_id": None, "sessions": {}}

    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                data = {"active_session_id": None, "sessions": {}}
            if "sessions" not in data or not isinstance(data["sessions"], dict):
                data["sessions"] = {}
            return data
    except Exception as exc:
        logger.warning("[session_manager] Could not read registry file: %s. Reinitializing.", exc)
        return {"active_session_id": None, "sessions": {}}


def save_registry(registry: Dict[str, Any]) -> None:
    """Saves the sessions registry to disk."""
    _ensure_dirs()
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error("[session_manager] Failed to save registry: %s", exc)


def create_session(title: str = "New Chat", filename: Optional[str] = None) -> Dict[str, Any]:
    """Creates a new isolated chat session."""
    registry = load_registry()
    session_id = str(uuid.uuid4())[:8]

    # Create session database directory
    db_dir = get_session_db_dir(session_id)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    new_session: Dict[str, Any] = {
        "id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "filename": filename,
        "chunk_count": 0,
        "db_ready": False,
        "chat_history": [],
    }

    registry["sessions"][session_id] = new_session
    registry["active_session_id"] = session_id
    save_registry(registry)

    logger.info("[session_manager] Created new session '%s' (ID: %s)", title, session_id)
    return new_session


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a session dict by ID."""
    registry = load_registry()
    return registry.get("sessions", {}).get(session_id)


def update_session(session: Dict[str, Any]) -> None:
    """Updates an existing session in the registry."""
    registry = load_registry()
    session_id = session.get("id")
    if not session_id:
        return
    session["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    registry["sessions"][session_id] = session
    save_registry(registry)


def set_active_session_id(session_id: str) -> None:
    """Sets the active session ID in registry."""
    registry = load_registry()
    if session_id in registry.get("sessions", {}):
        registry["active_session_id"] = session_id
        save_registry(registry)


def get_active_session_id() -> str:
    """Returns the current active session ID, or creates one if none exists."""
    registry = load_registry()
    active_id = registry.get("active_session_id")
    sessions = registry.get("sessions", {})

    if active_id and active_id in sessions:
        return active_id

    # If sessions exist, pick the first
    if sessions:
        first_id = list(sessions.keys())[0]
        registry["active_session_id"] = first_id
        save_registry(registry)
        return first_id

    # Otherwise create a fresh session
    new_sess = create_session(title="Chat 1")
    return new_sess["id"]


def list_sessions() -> List[Dict[str, Any]]:
    """Returns all sessions ordered by updated_at / created_at (most recent first)."""
    registry = load_registry()
    sessions = list(registry.get("sessions", {}).values())
    sessions.sort(key=lambda s: s.get("updated_at", s.get("created_at", "")), reverse=True)
    return sessions


def delete_session(session_id: str) -> str:
    """
    Deletes a session and its isolated vectorstore directory from disk.
    Returns the new active session ID.
    """
    registry = load_registry()
    sessions = registry.get("sessions", {})

    if session_id in sessions:
        del sessions[session_id]
        logger.info("[session_manager] Deleted session '%s'", session_id)

    # Delete disk vectorstore folder for this session
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info("[session_manager] Removed session directory: %s", session_dir)
        except Exception as exc:
            logger.warning("[session_manager] Could not delete directory %s: %s", session_dir, exc)

    # Determine new active session
    if sessions:
        new_active = list(sessions.keys())[0]
    else:
        new_sess = {
            "id": str(uuid.uuid4())[:8],
            "title": "Chat 1",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "filename": None,
            "chunk_count": 0,
            "db_ready": False,
            "chat_history": [],
        }
        sessions[new_sess["id"]] = new_sess
        new_active = new_sess["id"]

    registry["active_session_id"] = new_active
    save_registry(registry)
    return new_active


def clear_session_db(session_id: str) -> None:
    """Clears the vector store and document data for a specific session."""
    session_dir = get_session_db_dir(session_id)
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir, ignore_errors=True)
            os.makedirs(session_dir, exist_ok=True)
        except Exception as exc:
            logger.warning("[session_manager] Could not clear session dir %s: %s", session_dir, exc)

    session = get_session(session_id)
    if session:
        session["filename"] = None
        session["chunk_count"] = 0
        session["db_ready"] = False
        update_session(session)
        logger.info("[session_manager] Cleared database for session '%s'", session_id)


def auto_title(first_query: str) -> str:
    """Generates a concise 3-5 word title from the user's first query."""
    clean = first_query.strip().strip('"').strip("'")
    words = clean.split()
    if len(words) <= 5:
        return clean.capitalize()
    return " ".join(words[:5]).capitalize() + "…"


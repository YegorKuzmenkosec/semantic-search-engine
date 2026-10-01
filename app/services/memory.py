import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.models.schemas import SessionInfo


def _session_path(session_id: str) -> Path:
    settings.ensure_dirs()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)[:128]
    return settings.memory_path / f"{safe}.json"


def _load(session_id: str) -> dict:
    path = _session_path(session_id)
    if not path.exists():
        return {
            "session_id": session_id,
            "messages": [],
            "preferences": {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save(session_id: str, data: dict) -> None:
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    _session_path(session_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_history(session_id: str) -> list[dict[str, str]]:
    data = _load(session_id)
    return [{"role": m["role"], "content": m["content"]} for m in data.get("messages", [])]


def append_turn(session_id: str, question: str, answer: str) -> None:
    data = _load(session_id)
    data.setdefault("messages", [])
    data["messages"].extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    data["messages"] = data["messages"][-20:]
    _save(session_id, data)


def get_preferences(session_id: str) -> dict:
    return _load(session_id).get("preferences", {})


def update_preferences(session_id: str, updates: dict) -> dict:
    data = _load(session_id)
    prefs = data.setdefault("preferences", {})
    for key, value in updates.items():
        if value is not None:
            prefs[key] = value
    _save(session_id, data)
    return prefs


def get_session_info(session_id: str) -> SessionInfo:
    data = _load(session_id)
    return SessionInfo(
        session_id=session_id,
        message_count=len(data.get("messages", [])),
        preferences=data.get("preferences", {}),
        updated_at=datetime.fromisoformat(data["updated_at"]),
    )

"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class _OpenRequest:
    user_id: str
    text: str
    started_at: float
    timestamp: str


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, _OpenRequest] = {}
        self._latest_open_by_user: dict[str, str] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and return the correlation ID for its output record."""
        key = request_id or f"{user_id}:{uuid.uuid4().hex}"
        self._open[key] = _OpenRequest(
            user_id=user_id,
            text=text,
            started_at=time.perf_counter(),
            timestamp=utc_now_iso(),
        )
        self._latest_open_by_user[user_id] = key
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, and elapsed request time."""
        key = request_id or self._latest_open_by_user.get(user_id, user_id)
        opened = self._open.pop(key, None)
        if self._latest_open_by_user.get(user_id) == key:
            self._latest_open_by_user.pop(user_id, None)
        latency_ms = (
            round((time.perf_counter() - opened.started_at) * 1000, 3)
            if opened
            else None
        )
        self.logs.append(
            {
                "timestamp": opened.timestamp if opened else utc_now_iso(),
                "user_id": user_id,
                "request_id": key,
                "input": opened.text if opened else None,
                "output": text,
                "blocked": blocked,
                "layer": layer,
                "latency_ms": latency_ms,
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

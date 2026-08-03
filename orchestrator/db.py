from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from orchestrator.models import ChangeStatus, RiskLevel, TaskSpec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OrchestratorDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS changes (
                    change_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    branch_name TEXT,
                    patch_path TEXT,
                    diff_text TEXT,
                    changed_paths_json TEXT NOT NULL DEFAULT '[]',
                    validation_summary TEXT,
                    validation_log_path TEXT,
                    rejection_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    change_id TEXT NOT NULL,
                    actor_user_id INTEGER,
                    decision TEXT NOT NULL,
                    reason TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def create_change(self, change_id: str, task: TaskSpec) -> None:
        now = _now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO changes (
                    change_id, task_id, title, description, risk_level, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    change_id,
                    task.task_id,
                    task.title,
                    task.description,
                    task.risk_level.value,
                    ChangeStatus.QUEUED.value,
                    now,
                    now,
                ),
            )

    def update_change(self, change_id: str, **values: object) -> None:
        allowed = {
            "status",
            "branch_name",
            "patch_path",
            "diff_text",
            "changed_paths_json",
            "validation_summary",
            "validation_log_path",
            "rejection_reason",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported columns: {sorted(unknown)}")
        if not values:
            return
        values["updated_at"] = _now()
        columns = ", ".join(f"{name} = ?" for name in values)
        params = [
            value.value if isinstance(value, (ChangeStatus, RiskLevel)) else value
            for value in values.values()
        ]
        params.append(change_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE changes SET {columns} WHERE change_id = ?", params)

    def get_change(self, change_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM changes WHERE change_id = ?", (change_id,)).fetchone()
        return dict(row) if row else None

    def list_changes(self, status: ChangeStatus | None = None, limit: int = 20) -> list[dict]:
        query = "SELECT * FROM changes"
        params: list[object] = []
        if status:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def record_decision(
        self,
        change_id: str,
        decision: str,
        actor_user_id: int | None,
        reason: str | None = None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions (change_id, actor_user_id, decision, reason, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (change_id, actor_user_id, decision, reason, _now()),
            )

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_setting(self, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else None

    @staticmethod
    def encode_paths(paths: list[str]) -> str:
        return json.dumps(paths, ensure_ascii=False)

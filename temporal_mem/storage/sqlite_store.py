# temporal_mem/storage/sqlite_store.py

import json
import os
import sqlite3

from ..models import MemoryModel


class SqliteStore:
    """
    SQLite-based metadata store for MemoryModel.

    Day 3: fully working implementation.
    """

    def __init__(self, path: str = "~/.temporal_mem/history.db") -> None:
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory TEXT NOT NULL,
                type TEXT,
                slot TEXT,
                status TEXT,
                created_at TEXT,
                valid_until TEXT,
                decay_half_life_days INTEGER,
                confidence REAL,
                supersedes TEXT,
                source_turn_id TEXT,
                extra TEXT
            );
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_user_slot_status ON memories(user_id, slot, status);"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status);")
        self.conn.commit()

    @staticmethod
    def _row_to_model(row: sqlite3.Row) -> MemoryModel:
        return MemoryModel(
            id=row["id"],
            user_id=row["user_id"],
            memory=row["memory"],
            type=row["type"],
            slot=row["slot"],
            status=row["status"],
            created_at=row["created_at"],
            valid_until=row["valid_until"],
            decay_half_life_days=row["decay_half_life_days"],
            confidence=row["confidence"] if row["confidence"] is not None else 0.0,
            supersedes=json.loads(row["supersedes"]) if row["supersedes"] else [],
            source_turn_id=row["source_turn_id"],
            extra=json.loads(row["extra"]) if row["extra"] else {},
        )

    def insert(self, mem: MemoryModel) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO memories (
                id,
                user_id,
                memory,
                type,
                slot,
                status,
                created_at,
                valid_until,
                decay_half_life_days,
                confidence,
                supersedes,
                source_turn_id,
                extra
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem.id,
                mem.user_id,
                mem.memory,
                mem.type,
                mem.slot,
                mem.status,
                mem.created_at,
                mem.valid_until,
                mem.decay_half_life_days,
                mem.confidence,
                json.dumps(mem.supersedes or []),
                mem.source_turn_id,
                json.dumps(mem.extra or {}),
            ),
        )
        self.conn.commit()

    def get_by_id(self, mem_id: str) -> MemoryModel | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM memories WHERE id = ? LIMIT 1;", (mem_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_model(row)

    def update_status(self, mem_id: str, new_status: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE memories SET status = ? WHERE id = ?;",
            (new_status, mem_id),
        )
        self.conn.commit()

    def get_active_by_slot(self, user_id: str, slot: str) -> list[MemoryModel]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ?
              AND slot = ?
              AND status = 'active';
            """,
            (user_id, slot),
        )
        rows = cur.fetchall()
        return [self._row_to_model(r) for r in rows]

    def list_by_user(self, user_id: str, status: str = "active") -> list[MemoryModel]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ?
              AND status = ?
            ORDER BY datetime(created_at) DESC;
            """,
            (user_id, status),
        )
        rows = cur.fetchall()
        return [self._row_to_model(r) for r in rows]

    def list_by_ids(self, ids: list[str]) -> list[MemoryModel]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT * FROM memories
            WHERE id IN ({placeholders});
            """,
            ids,
        )
        rows = cur.fetchall()
        return [self._row_to_model(r) for r in rows]

    def close(self) -> None:
        self.conn.close()

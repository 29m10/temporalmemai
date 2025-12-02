# temporal_mem/memory.py

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import builtins

from .embedding.openai_embedder import OpenAIEmbedder  # still unused on Day 3
from .llm.extractor import FactExtractor
from .models import MemoryModel  # noqa: TC001
from .storage.qdrant_store import QdrantStore  # still unused on Day 3
from .storage.sqlite_store import SqliteStore
from .temporal.engine import TemporalEngine


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class Memory:
    """
    Public facade.

    Day 3:
    - add() uses FactExtractor + TemporalEngine + SqliteStore
    - list() reads from SqliteStore
    - search/update/delete are still mostly stubs (for later days)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}

        sqlite_path = config.get("sqlite_path", "~/.temporal_mem/history.db")
        qdrant_host = config.get("qdrant_host", "localhost")
        qdrant_port = int(config.get("qdrant_port", 6333))
        qdrant_collection = config.get("qdrant_collection", "temporal_mem_default")

        openai_api_key = config.get("openai_api_key")
        embed_model = config.get("embed_model", "text-embedding-3-small")
        llm_model = config.get("llm_model", "gpt-4.1-mini")
        llm_temp = float(config.get("llm_temperature", 0.0))

        # Core components
        self.metadata_store = SqliteStore(path=sqlite_path)
        self.temporal_engine = TemporalEngine(metadata_store=self.metadata_store)
        self.fact_extractor = FactExtractor(
            api_key=openai_api_key,
            model=llm_model,
            temperature=llm_temp,
        )

        # These will be used later (Day 4) for search
        self.embedder = OpenAIEmbedder(
            api_key=openai_api_key,
            model=embed_model,
        )
        self.vector_store = QdrantStore(
            host=qdrant_host,
            port=qdrant_port,
            collection=qdrant_collection,
            vector_size=1536,
        )

    # ------------------------------------------------------------------ #
    # ADD (Day 3)
    # ------------------------------------------------------------------ #

    def add(
        self,
        messages: str | builtins.list[dict[str, str]],
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        v1 (Day 3):
        - Extract facts using FactExtractor
        - Convert to MemoryModel using TemporalEngine
        - Store in SQLite via SqliteStore

        No embeddings / Qdrant yet.
        """
        if isinstance(messages, str):
            msg_list = [{"role": "user", "content": messages}]
        else:
            msg_list = messages

        source_turn_id = metadata.get("turn_id") if metadata else None

        # 1. extract facts
        fact_candidates = self.fact_extractor.extract_from_messages(msg_list)

        # 2. temporal engine -> MemoryModel
        mem_models = self.temporal_engine.process_write_batch(
            facts=fact_candidates,
            user_id=user_id,
            source_turn_id=source_turn_id,
        )

        # 3. store in SQLite
        for mem in mem_models:
            if not mem.created_at:
                mem.created_at = _now_iso()
            self.metadata_store.insert(mem)

        return {
            "results": [self._serialize_memory(m) for m in mem_models],
        }

    # ------------------------------------------------------------------ #
    # LIST (Day 3)
    # ------------------------------------------------------------------ #

    def list(
        self,
        user_id: str,
        status: str = "active",
    ) -> dict[str, Any]:
        """
        v1 (Day 3):
        - Read memories from SQLite by user + status.
        """
        memories = self.metadata_store.list_by_user(user_id, status=status)
        return {
            "results": [self._serialize_memory(m) for m in memories],
        }

    # ------------------------------------------------------------------ #
    # STUBS (to be filled later)
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        user_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Day 3: semantic search not implemented yet.
        """
        return {"results": []}

    def delete(self, memory_id: str) -> None:
        """
        Day 3: stub; will implement soft-delete + Qdrant removal later.
        """
        return

    def update(self, memory_id: str, new_content: str) -> dict[str, Any] | None:
        """
        Day 3: stub; will implement "archive old + insert new" later.
        """
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialize_memory(mem: MemoryModel) -> dict[str, Any]:
        return {
            "id": mem.id,
            "user_id": mem.user_id,
            "memory": mem.memory,
            "type": mem.type,
            "slot": mem.slot,
            "status": mem.status,
            "created_at": mem.created_at,
            "valid_until": mem.valid_until,
            "decay_half_life_days": mem.decay_half_life_days,
            "confidence": mem.confidence,
            "supersedes": mem.supersedes,
            "source_turn_id": mem.source_turn_id,
            "extra": mem.extra,
        }

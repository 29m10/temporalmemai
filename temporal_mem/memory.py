# temporal_mem/memory.py

from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .llm.extractor import FactExtractor
from .temporal.engine import TemporalEngine
from .storage.sqlite_store import SqliteStore
from .storage.qdrant_store import QdrantStore
from .embedding.openai_embedder import OpenAIEmbedder
from .models import MemoryModel


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _parse_iso_maybe(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    # handle 2025-01-01T00:00:00Z and 2025-01-01T00:00:00
    s = dt_str.replace("Z", "")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


class Memory:
    """
    Public facade.

    Day 4:
    - add() uses FactExtractor + TemporalEngine + SqliteStore + Qdrant indexing
    - list() reads from SqliteStore
    - search() does:
        query -> embedding -> Qdrant search -> SQLite fetch -> temporal-aware scoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
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

        # Embedding + vector store (Day 4)
        self.embedder = OpenAIEmbedder(
            api_key=openai_api_key,
            model=embed_model,
        )
        # Vector size depends on model; for text-embedding-3-small it's 1536
        self.vector_store = QdrantStore(
            host=qdrant_host,
            port=qdrant_port,
            collection=qdrant_collection,
            vector_size=1536,
        )

    # ------------------------------------------------------------------ #
    # ADD
    # ------------------------------------------------------------------ #

    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        v2 (Day 4):
        - Extract facts using FactExtractor
        - Convert to MemoryModel using TemporalEngine
        - Store metadata in SQLite
        - Compute embeddings and index in Qdrant
        """
        if isinstance(messages, str):
            msg_list = [{"role": "user", "content": messages}]
        else:
            msg_list = messages

        source_turn_id = metadata.get("turn_id") if metadata else None

        # 1. extract facts
        fact_candidates = self.fact_extractor.extract_from_messages(msg_list)

        if not fact_candidates:
            return {"results": []}

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

        # 4. index in Qdrant (embeddings)
        # Only index active records
        for mem in mem_models:
            if mem.status != "active":
                continue
            try:
                vec = self.embedder.embed_one(mem.memory)
            except Exception as e:
                print("[Memory.add] Embedding failed for mem.id =", mem.id, "err:", e)
                traceback.format_exc()
                continue

            payload = {
                "user_id": mem.user_id,
                "type": mem.type,
                "slot": mem.slot,
                "status": mem.status,
                "created_at": mem.created_at,
                "valid_until": mem.valid_until,
                "confidence": mem.confidence,
            }

            try:
                self.vector_store.upsert_point(
                    memory_id=mem.id,
                    vector=vec,
                    payload=payload,
                )
            except Exception:
                # Qdrant failure should not break add(); metadata already stored
                continue

        return {
            "results": [self._serialize_memory(m) for m in mem_models],
        }

    # ------------------------------------------------------------------ #
    # LIST
    # ------------------------------------------------------------------ #

    def list(
        self,
        user_id: str,
        status: str = "active",
    ) -> Dict[str, Any]:
        """
        v1:
        - Read memories from SQLite by user + status.
        """
        memories = self.metadata_store.list_by_user(user_id, status=status)
        return {
            "results": [self._serialize_memory(m) for m in memories],
        }

    # ------------------------------------------------------------------ #
    # SEARCH
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Semantic search over user's memories.

        Steps:
        - Embed query
        - Vector search in Qdrant (filtered by user_id and optional filters)
        - Fetch MemoryModel from SQLite by ids
        - Temporal-aware re-ranking
        """
        filters = filters or {}
        if "status" not in filters:
            filters["status"] = "active"

        try:
            q_vec = self.embedder.embed_one(query)
        except Exception as e:
            print("[Memory.search] Embedding failed for query:", query, "err:", e)
            traceback.print_exc()
            return {"results": []}

        try:
            vec_results = self.vector_store.search(
                query_vector=q_vec,
                user_id=user_id,
                limit=limit,
                filters=filters,
            )
        except Exception as e:
            print("[Memory.search] Qdrant search failed for query:", query, "err:", e)
            traceback.print_exc()
            return {"results": []}

        ids = [r["id"] for r in vec_results]
        if not ids:
            return {"results": []}

        mems = self.metadata_store.list_by_ids(ids)
        mem_by_id = {m.id: m for m in mems}

        # Merge similarity score from Qdrant with temporal info for ranking
        now = datetime.utcnow()
        combined: List[Dict[str, Any]] = []

        for r in vec_results:
            mem_id = r["id"]
            score = r["score"]
            mem = mem_by_id.get(mem_id)
            if not mem:
                continue

            final_score = self._compute_rank_score(
                base_score=score,
                mem=mem,
                now=now,
            )

            combined.append(
                {
                    "memory": self._serialize_memory(mem),
                    "similarity": score,
                    "score": final_score,
                }
            )

        # Sort by final score descending
        combined.sort(key=lambda x: x["score"], reverse=True)

        return {"results": combined}

    def _compute_rank_score(
        self,
        base_score: float,
        mem: MemoryModel,
        now: datetime,
    ) -> float:
        """
        Simple temporal-aware ranking.

        Start from base_score (similarity) and adjust:
        - penalize if memory is expired (beyond valid_until)
        - slight penalty if type is temp_state and old
        - slight bonus for profile_fact / preference
        """
        score = base_score

        valid_until_dt = _parse_iso_maybe(mem.valid_until)
        created_at_dt = _parse_iso_maybe(mem.created_at)

        # Expiry penalty
        if valid_until_dt and now > valid_until_dt:
            # expired memories get a heavy penalty
            score -= 0.5

        # Type based adjustments
        if mem.type == "profile_fact":
            score += 0.1
        elif mem.type == "preference":
            score += 0.05
        elif mem.type == "temp_state":
            # Newer temp states preferred over older ones
            if created_at_dt:
                age_days = (now - created_at_dt).days
                if age_days > 7:
                    score -= 0.1
        elif mem.type == "episodic_event":
            # mild penalty for very old events
            if created_at_dt:
                age_days = (now - created_at_dt).days
                if age_days > 30:
                    score -= 0.05

        # Confidence adjustment
        if mem.confidence < 0.5:
            score -= 0.2
        elif mem.confidence > 0.9:
            score += 0.05

        return score

    # ------------------------------------------------------------------ #
    # STUBS (for future days)
    # ------------------------------------------------------------------ #

    def delete(self, memory_id: str) -> None:
        """
        Soft-delete in SQLite + remove from Qdrant.
        """
        # v1: mark as deleted in SQLite, best-effort Qdrant delete
        existing = self.metadata_store.get_by_id(memory_id)
        if not existing:
            return
        self.metadata_store.update_status(memory_id, "deleted")
        try:
            self.vector_store.delete(memory_id)
        except Exception as e:
            print("[Memory.delete] Qdrant delete failed for memory_id:", memory_id, "err:", e)
            traceback.print_exc()
            pass

    def update(self, memory_id: str, new_content: str) -> Optional[Dict[str, Any]]:
        """
        Simple update pattern:
        - archive old memory
        - create new memory with same type/slot/user and new text
        - reindex new memory
        """
        old = self.metadata_store.get_by_id(memory_id)
        if not old:
            return None

        # Archive old
        self.metadata_store.update_status(memory_id, "archived")

        # Create new memory model
        new_mem = MemoryModel(
            id=memory_id,  # could also generate a new id if you prefer
            user_id=old.user_id,
            memory=new_content,
            type=old.type,
            slot=old.slot,
            status="active",
            created_at=_now_iso(),
            valid_until=old.valid_until,
            decay_half_life_days=old.decay_half_life_days,
            confidence=old.confidence,
            supersedes=[memory_id],
            source_turn_id=old.source_turn_id,
            extra=old.extra,
        )

        self.metadata_store.insert(new_mem)

        # Reindex in Qdrant
        try:
            vec = self.embedder.embed_one(new_content)
            payload = {
                "user_id": new_mem.user_id,
                "type": new_mem.type,
                "slot": new_mem.slot,
                "status": new_mem.status,
                "created_at": new_mem.created_at,
                "valid_until": new_mem.valid_until,
                "confidence": new_mem.confidence,
            }
            self.vector_store.upsert_point(
                memory_id=new_mem.id,
                vector=vec,
                payload=payload,
            )
        except Exception as e:
            print("[Memory.update] Qdrant upsert failed for memory_id:", memory_id, "err:", e)
            traceback.print_exc()
            pass

        return self._serialize_memory(new_mem)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialize_memory(mem: MemoryModel) -> Dict[str, Any]:
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
    
    def reindex_user(self, user_id: str, status: str = "active") -> Dict[str, int]:
        """
        Rebuild Qdrant index for all memories of a user from SQLite.

        - Reads all memories for user_id (and status)
        - Embeds each memory text
        - Upserts into Qdrant

        Returns: {"total": X, "indexed": Y, "failed": Z}
        """
        mems = self.metadata_store.list_by_user(user_id, status=status)
        total = len(mems)
        indexed = 0
        failed = 0

        for mem in mems:
            try:
                vec = self.embedder.embed_one(mem.memory)
            except Exception as e:
                print(f"[reindex_user] Embedding failed for {mem.id}: {e}")
                failed += 1
                continue

            payload = {
                "user_id": mem.user_id,
                "type": mem.type,
                "slot": mem.slot,
                "status": mem.status,
                "created_at": mem.created_at,
                "valid_until": mem.valid_until,
                "confidence": mem.confidence,
            }

            try:
                self.vector_store.upsert_point(
                    memory_id=mem.id,
                    vector=vec,
                    payload=payload,
                )
                indexed += 1
            except Exception as e:
                print(f"[reindex_user] Upsert failed for {mem.id}: {e}")
                failed += 1

        return {"total": total, "indexed": indexed, "failed": failed}


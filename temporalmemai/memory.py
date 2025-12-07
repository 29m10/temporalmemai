# temporalmemai/memory.py

from __future__ import annotations

import os
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import builtins

from .embedding.openai_embedder import OpenAIEmbedder
from .llm.extractor import FactExtractor
from .models import MemoryModel
from .rerankers.factory import create_reranker
from .storage.qdrant_store import QdrantStore
from .storage.sqlite_store import SqliteStore
from .temporal.engine import TemporalEngine


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _parse_iso_maybe(dt_str: str | None) -> datetime | None:
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

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}

        # -------------------------------------------------------
        # SQLite
        # -------------------------------------------------------
        sqlite_path = config.get("sqlite_path") or os.getenv("SQLITE_PATH")
        if not sqlite_path:
            raise ValueError("SQLite path not provided. Set SQLITE_PATH or pass sqlite_path.")

        # -------------------------------------------------------
        # OpenAI credentials
        # -------------------------------------------------------
        openai_api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key missing. Set OPENAI_API_KEY or provide `openai_api_key`."
            )

        embed_model = (
            config.get("embed_model") or os.getenv("OPENAI_EMBED_MODEL") or "text-embedding-3-small"
        )

        llm_model = config.get("llm_model") or os.getenv("OPENAI_LLM_MODEL") or "gpt-4.1-mini"

        temp_str = (
            str(config.get("llm_temperature"))
            if config.get("llm_temperature") is not None
            else os.getenv("OPENAI_LLM_TEMPERATURE")
        )
        try:
            llm_temp = float(temp_str) if temp_str is not None else 0.0
        except Exception:
            llm_temp = 0.0

        # -------------------------------------------------------
        # Qdrant configuration
        # Cloud: QDRANT_URL + QDRANT_API_KEY
        # Local: QDRANT_HOST + QDRANT_PORT
        # -------------------------------------------------------
        qdrant_url = config.get("qdrant_url") or os.getenv("QDRANT_URL")
        qdrant_api_key = config.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY")

        qdrant_host = config.get("qdrant_host") or os.getenv("QDRANT_HOST")
        qdrant_port = config.get("qdrant_port") or os.getenv("QDRANT_PORT")

        # At least one connection method must be provided
        if not qdrant_url and not qdrant_host:
            raise ValueError(
                "Qdrant config missing. Provide either QDRANT_URL (cloud) or QDRANT_HOST (local)."
            )

        # If URL is provided, API key is required
        if qdrant_url and not qdrant_api_key:
            raise ValueError(
                "Qdrant API key required when using QDRANT_URL. "
                "Set QDRANT_API_KEY or provide qdrant_api_key."
            )

        # If HOST is provided, PORT is required
        if qdrant_host and not qdrant_port:
            raise ValueError(
                "Qdrant port required when using QDRANT_HOST. "
                "Set QDRANT_PORT or provide qdrant_port."
            )

        # Collection name defaults to "temporalmemai_default" if not provided
        collection_name = (
            config.get("qdrant_collection")
            or os.getenv("QDRANT_COLLECTION")
            or "temporalmemai_default"
        )

        # -------------------------------------------------------
        # Initialize components
        # -------------------------------------------------------
        self.metadata_store = SqliteStore(path=sqlite_path)
        self.temporal_engine = TemporalEngine(self.metadata_store)

        # LLM extractor
        self.fact_extractor = FactExtractor(
            api_key=openai_api_key,
            model=llm_model,
            temperature=llm_temp,
        )

        # Embeddings
        self.embedder = OpenAIEmbedder(
            api_key=openai_api_key,
            model=embed_model,
        )

        # Vector store (Qdrant)
        self.vector_store = QdrantStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            host=qdrant_host,
            port=qdrant_port,
            collection=collection_name,
            vector_size=self.embedder.vector_size,
        )

        # Reranker
        reranker_cfg = config.get("reranker")
        self.reranker = create_reranker(reranker_cfg)

    # ------------------------------------------------------------------ #
    # Lazy-expire helper (per user, easy to remove later)
    # ------------------------------------------------------------------ #

    def _lazy_expire_user(self, user_id: str) -> None:
        """
        Best-effort lazy expiration for a single user.

        - Calls metadata_store.expire_user_memories(user_id).
        - Used in add(), list(), and search() so that:
          - by the time we look at "active" memories,
            anything past valid_until is marked "expired".

        To remove this behavior in future:
        - delete this method
        - remove calls to _lazy_expire_user in add(), list(), search().
        """
        try:
            expired = self.metadata_store.expire_user_memories(user_id)
            if expired:
                print(f"[Memory] Lazy-expired {expired} memories for user={user_id}")
        except Exception as e:
            print(f"[Memory] Lazy expire failed for user={user_id}: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # ADD
    # ------------------------------------------------------------------ #

    def add(
        self,
        messages: str | builtins.list[dict[str, str]],
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Add memories from a message or list of chat messages.

        Pipeline:
        0. Lazy-expire outdated memories for this user in SQLite.
        1. Extract fact candidates via FactExtractor (LLM).
        2. TemporalEngine converts them to MemoryModel objects
           (type, slot, TTL, etc.).
        3. Store all memories in SQLite (source of truth).
        4. For each ACTIVE memory:
           - Embed text
           - Upsert into Qdrant with payload (user_id, type, slot, status, ...)

        This guarantees: once add() returns successfully, future search()
        calls (in any process) can retrieve these memories, as long as
        Qdrant data persists.
        """
        # Lazy expire before we add more context for this user
        self._lazy_expire_user(user_id)

        if isinstance(messages, str):
            msg_list = [{"role": "user", "content": messages}]
        else:
            msg_list = messages

        source_turn_id = metadata.get("turn_id") if metadata else None

        # 1. Fact extraction
        fact_candidates = self.fact_extractor.extract_from_messages(msg_list)
        print(f"[Memory.add] Extracted {len(fact_candidates)} fact candidates")

        if not fact_candidates:
            return {"results": []}

        # 2. Temporal engine -> MemoryModel
        mem_models = self.temporal_engine.process_write_batch(
            facts=fact_candidates,
            user_id=user_id,
            source_turn_id=source_turn_id,
        )
        print(f"[Memory.add] Temporal engine produced {len(mem_models)} memories")

        # 3. Store in SQLite
        for mem in mem_models:
            if not mem.created_at:
                mem.created_at = _now_iso()
            self.metadata_store.insert(mem)

        # 4. Index active memories in Qdrant
        indexed = 0
        for mem in mem_models:
            if mem.status != "active":
                continue

            try:
                vec = self.embedder.embed_one(mem.memory)
            except Exception as e:
                print(f"[Memory.add] Embedding failed for {mem.id}: {e}")
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
                print(f"[Memory.add] Qdrant upsert failed for {mem.id}: {e}")
                continue

        print(f"[Memory.add] Indexed {indexed} active memories into Qdrant")

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
    ) -> dict[str, Any]:
        """
        v1:
        - Lazy-expire this user's memories.
        - Read memories from SQLite by user + status.
        """
        self._lazy_expire_user(user_id)

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
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        rerank: bool = False,
    ) -> dict[str, Any]:
        """
        Semantic search over user's memories.

        If rerank=True AND a reranker is configured, we:
        - overfetch from Qdrant (limit*3)
        - rerank that pool
        - then apply temporal scoring on top

        If rerank=False OR no reranker configured:
        - we just use vector similarity + temporal scoring (no reranker).
        """
        filters = filters or {}
        if "status" not in filters:
            filters["status"] = "active"

        # 1) embed query
        try:
            q_vec = self.embedder.embed_one(query)
        except Exception as e:
            print("[Memory.search] Embedding failed:", e)
            traceback.print_exc()
            return {"results": []}

        # Do we actually have a reranker?
        use_reranker = bool(rerank and self.reranker)

        # 2) vector search
        try:
            raw_limit = limit * 3 if use_reranker else limit
            vec_results = self.vector_store.search(
                query_vector=q_vec,
                user_id=user_id,
                limit=raw_limit,
                filters=filters,
            )
        except Exception as e:
            print("[Memory.search] Qdrant search failed:", e)
            traceback.print_exc()
            return {"results": []}

        if not vec_results:
            return {"results": []}

        ids = [r["id"] for r in vec_results]
        mems = self.metadata_store.list_by_ids(ids)
        mem_by_id = {m.id: m for m in mems}

        # Build candidate docs from vector search
        candidates = []
        for r in vec_results:
            mem = mem_by_id.get(r["id"])
            if not mem:
                continue
            candidates.append(
                {
                    "id": mem.id,
                    "memory": mem.memory,
                    "vector_score": r["score"],
                }
            )

        # 3) Optional rerank
        if use_reranker:
            try:
                candidates = self.reranker.rerank(
                    query=query,
                    documents=candidates,  # <-- fixed, not documents[candidates]
                    top_k=limit,
                )
            except Exception as e:
                print("[Memory.search] Reranker failed:", e)
                traceback.print_exc()
                # fall back to vector-only order (already in candidates)

        # 4) Merge with temporal scoring + serialize
        now = datetime.utcnow()
        results = []

        # we already limited via top_k in reranker, but still slice defensively
        for c in candidates[:limit]:
            mem = mem_by_id.get(c["id"])
            if not mem:
                continue

            base_score = c.get("vector_score", 0.0)
            rerank_score = c.get("rerank_score", None)

            # Combine scores: tune this if needed
            combined_score = base_score
            if rerank_score is not None:
                combined_score = 0.2 * base_score + 0.8 * rerank_score

            final_score = self._compute_rank_score(
                base_score=combined_score,
                mem=mem,
                now=now,
            )

            results.append(
                {
                    "memory": self._serialize_memory(mem),
                    "similarity": base_score,
                    "rerank_score": rerank_score,
                    "score": final_score,
                }
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return {"results": results}

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

        # Expiry penalty (extra safety; lazy expire should already handle this)
        if valid_until_dt and now > valid_until_dt:
            # expired memories get a heavy penalty
            score -= 0.5

        # Type based adjustments
        if mem.type == "profile_fact":
            score += 0.1
        elif mem.type == "preference":
            score += 0.05
        elif mem.type == "temp_state" and created_at_dt and (now - created_at_dt).days > 7:
            # Newer temp states preferred over older ones
            score -= 0.1
        elif mem.type == "episodic_event" and created_at_dt and (now - created_at_dt).days > 30:
            # mild penalty for very old events
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

    def update(self, memory_id: str, new_content: str) -> dict[str, Any] | None:
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
            kind=old.kind,
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

        return self._serialize_memory(new_mem)

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
            "kind": mem.kind,
            "status": mem.status,
            "created_at": mem.created_at,
            "valid_until": mem.valid_until,
            "decay_half_life_days": mem.decay_half_life_days,
            "confidence": mem.confidence,
            "supersedes": mem.supersedes,
            "source_turn_id": mem.source_turn_id,
            "extra": mem.extra,
        }

    def reindex_user(self, user_id: str, status: str = "active") -> dict[str, int]:
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

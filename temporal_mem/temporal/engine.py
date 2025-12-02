# temporal_mem/temporal/engine.py

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import uuid4

from ..models import MemoryModel, FactCandidate
from ..storage.sqlite_store import SqliteStore


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class TemporalEngine:
    """
    Responsible for:
    - Mapping FactCandidate -> MemoryModel (type, TTL, decay)
    - Conflict resolution (superseding old slot memories)
    - Temporal filtering + ranking on read.
    """

    def __init__(self, metadata_store: SqliteStore) -> None:
        self.metadata_store = metadata_store

    # ------------------------------------------------------------------ #
    # Mapping & policies
    # ------------------------------------------------------------------ #

    def _map_category_to_type(self, category: str) -> str:
        match category:
            case "profile":
                return "profile_fact"
            case "preference":
                return "preference"
            case "event":
                return "episodic_event"
            case "temp_state":
                return "temp_state"
            case _:
                return "other"

    def _apply_policies(self, mem: MemoryModel) -> MemoryModel:
        """
        Set valid_until / decay_half_life_days based on type/slot.
        v1: simple hard-coded rules.
        """
        if mem.type == "temp_state":
            # short-lived state
            mem.decay_half_life_days = 1
            mem.valid_until = (datetime.utcnow() + timedelta(days=3)).isoformat() + "Z"
        elif mem.type == "preference":
            mem.decay_half_life_days = 60
            mem.valid_until = None
        elif mem.type == "profile_fact":
            mem.decay_half_life_days = None
            mem.valid_until = None
        elif mem.type == "episodic_event":
            mem.decay_half_life_days = 7
            mem.valid_until = None
        else:
            mem.decay_half_life_days = 30
            mem.valid_until = None

        return mem

    def _resolve_conflicts(self, mem: MemoryModel) -> MemoryModel:
        """
        If this memory has a slot, archive previous active memories with same slot.
        """
        if not mem.slot:
            return mem

        existing = self.metadata_store.get_active_by_slot(mem.user_id, mem.slot)
        supersedes_ids: List[str] = []
        for old in existing:
            self.metadata_store.update_status(old.id, "archived")
            supersedes_ids.append(old.id)

        if supersedes_ids:
            mem.supersedes = supersedes_ids
        return mem

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def from_fact_candidate(
        self,
        fact: FactCandidate,
        user_id: str,
        source_turn_id: Optional[str] = None,
    ) -> MemoryModel:
        mem_type = self._map_category_to_type(fact.category)
        created_at = _now_iso()

        mem = MemoryModel(
            id=str(uuid4()),
            user_id=user_id,
            memory=fact.text,
            type=mem_type,
            slot=fact.slot,
            status="active",
            created_at=created_at,
            valid_until=None,
            decay_half_life_days=None,
            confidence=fact.confidence,
            supersedes=[],
            source_turn_id=source_turn_id,
            extra={},
        )

        mem = self._apply_policies(mem)
        mem = self._resolve_conflicts(mem)
        return mem

    def process_write_batch(
        self,
        facts: List[FactCandidate],
        user_id: str,
        source_turn_id: Optional[str] = None,
    ) -> List[MemoryModel]:
        """
        Turn a list of FactCandidate into enriched MemoryModel objects.
        v1:
        - Drop very low-confidence facts (<0.5)
        - Apply mapping + policies + conflict resolution
        """
        memories: List[MemoryModel] = []
        for fact in facts:
            if fact.confidence < 0.5:
                continue
            mem = self.from_fact_candidate(
                fact=fact,
                user_id=user_id,
                source_turn_id=source_turn_id,
            )
            memories.append(mem)
        return memories

    def filter_and_rank(
        self,
        memories: List[MemoryModel],
    ) -> List[MemoryModel]:
        """
        v1: just return as-is.

        Later:
        - drop expired
        - dedupe per slot
        - temporal scoring
        """
        return memories

# temporal_mem/temporal/engine.py

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

from ..models import FactCandidate, MemoryModel
from ..storage.sqlite_store import SqliteStore  # noqa: TC001


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

    def _apply_policies(self, mem: MemoryModel, fact: FactCandidate) -> MemoryModel:
        """
        Priority:
        1. duration_in_days from LLM (for temp stuff)
        2. fallback to type-based defaults
        """
        now = datetime.utcnow()

        # 1) explicit duration wins
        if fact.duration_in_days is not None and fact.duration_in_days > 0:
            days = fact.duration_in_days
            mem.valid_until = (now + timedelta(days=days)).isoformat() + "Z"
            # simple rule: half-life is half the duration (min 1 day)
            mem.decay_half_life_days = max(1, days // 2) or 1
            return mem

        # 2) fallback to type-based defaults
        if mem.type == "temp_state":
            mem.decay_half_life_days = 1
            mem.valid_until = (now + timedelta(days=3)).isoformat() + "Z"
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
        supersedes_ids: list[str] = []
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
        source_turn_id: str | None = None,
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
        return mem  # noqa: RET504

    def process_write_batch(
        self,
        facts: list[FactCandidate],
        user_id: str,
        source_turn_id: str | None = None,
    ) -> list[MemoryModel]:
        """
        Turn a list of FactCandidate into enriched MemoryModel objects.
        v1:
        - Drop very low-confidence facts (<0.5)
        - Apply mapping + policies + conflict resolution
        """
        memories: list[MemoryModel] = []
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
        memories: list[MemoryModel],
    ) -> list[MemoryModel]:
        """
        v1: just return as-is.

        Later:
        - drop expired
        - dedupe per slot
        - temporal scoring
        """
        return memories

    def _type_and_slot_from_fact(self, fact: FactCandidate) -> tuple[str, str | None]:
        """
        Decide internal memory type + slot from the fact.

        - Prefer fact.kind when present (home_location vs current_location).
        - Fall back to fact.slot + category mapping.
        """
        # Kind-based routing (more semantic)
        if fact.kind == "home_location":
            return "profile_fact", "home_location"

        if fact.kind == "current_location":
            # current location is a temp state by definition
            return "temp_state", "current_location"

        if fact.kind == "trip":
            return "episodic_event", "trip"

        # fallback: category + provided slot
        mem_type = self._map_category_to_type(fact.category)
        slot = fact.slot or None
        return mem_type, slot

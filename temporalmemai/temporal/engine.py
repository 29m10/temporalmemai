# temporalmemai/temporal/engine.py

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
        Decide valid_until + decay_half_life_days for a memory.

        Precedence:
        1) duration_minutes  -> now + minutes
        2) duration_hours    -> now + hours
        3) duration_in_days  -> now + days
        4) fallback by mem.type
        """
        now = datetime.utcnow()

        # 1) Minutes (most precise, e.g. "45 minutes", "20 minutes")
        minutes = getattr(fact, "duration_in_minutes", None)
        if minutes is not None and minutes > 0:
            mem.valid_until = (now + timedelta(minutes=minutes)).isoformat() + "Z"
            # For very short-lived states, TTL is the main guard; half-life can be 1 day.
            mem.decay_half_life_days = 1
            return mem

        # 2) Hours (e.g. "for 2 hours at Kolkata airport")
        hours = getattr(fact, "duration_in_hours", None)
        if hours is not None and hours > 0:
            mem.valid_until = (now + timedelta(hours=hours)).isoformat() + "Z"
            mem.decay_half_life_days = 1
            return mem

        # 3) Days (e.g. "for 3 days", "for a week")
        if fact.duration_in_days is not None and fact.duration_in_days > 0:
            days = fact.duration_in_days
            mem.valid_until = (now + timedelta(days=days)).isoformat() + "Z"
            mem.decay_half_life_days = max(1, days // 2) or 1
            return mem

        # 4) Fallback: type-based defaults
        if mem.type == "temp_state":
            # No explicit duration → short-lived by default
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
        Conflict resolution is disabled for now.

        Previously:
        - We archived existing active memories for the same user + slot.

        For now:
        - We keep ALL memories, even if they share the same slot
          (home_location, current_location, hobby, etc.).

        This makes the system append-only, which is simpler while we
        debug and refine semantics. We can reintroduce smarter conflict
        logic later.
        """
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
        # Use semantic routing (kind + slot)
        mem_type, slot = self._type_and_slot_from_fact(fact)
        created_at = _now_iso()

        mem = MemoryModel(
            id=str(uuid4()),
            user_id=user_id,
            memory=fact.text,
            type=mem_type,
            slot=slot,
            kind=fact.kind,
            status="active",
            created_at=created_at,
            valid_until=None,
            decay_half_life_days=None,
            confidence=fact.confidence,
            supersedes=[],
            source_turn_id=source_turn_id,
            extra={},
        )

        # 👇 IMPORTANT: pass both mem AND fact
        mem = self._apply_policies(mem, fact)
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

from pydantic import BaseModel


class FactCandidate(BaseModel):
    """
    Output of the fact extraction layer.

    - text: human-readable fact
    - category: coarse type ("profile", "preference", "event", "temp_state", "other")
    - slot: semantic slot ("location", "home_location", "current_location", "job", "hobby", etc.)
    - confidence: LLM's belief (0-1)
    - stability: "persistent" | "temporary" | "unknown"
    - temporal_scope: "now" | "today" | "this_week" | "this_month" | "specific_range" | "none"
    - kind: domain-specific sub-type ("home_location", "current_location", "trip", etc.)
    - duration_in_days: interpreted duration of this fact, if any (e.g. "two days" -> 2)
    """

    text: str
    category: str
    slot: str | None = None
    confidence: float = 1.0

    stability: str | None = None  # persistent | temporary | unknown
    temporal_scope: str | None = (
        None  # now | today | this_week | this_month | specific_range | none
    )
    kind: str | None = None  # home_location | current_location | trip | ...
    duration_in_days: int | None = None


class MemoryModel(BaseModel):
    id: str
    user_id: str
    memory: str
    type: str  # "profile_fact" | "preference" | "episodic_event" | "temp_state" | "task_state" | "other"
    slot: str | None = None
    status: str = "active"  # "active" | "archived" | "deleted"
    created_at: str
    valid_until: str | None = None
    decay_half_life_days: int | None = None
    confidence: float = 1.0
    supersedes: list[str] = []
    source_turn_id: str | None = None
    extra: dict = {}

from pydantic import BaseModel


class FactCandidate(BaseModel):
    text: str
    category: str  # "profile" | "preference" | "event" | "temp_state" | "other"
    slot: str | None = None
    confidence: float = 1.0  # 0.0-1.0


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

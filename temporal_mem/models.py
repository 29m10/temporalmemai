from typing import List, Optional, Dict
from pydantic import BaseModel


class FactCandidate(BaseModel):
    text: str
    category: str                  # "profile" | "preference" | "event" | "temp_state" | "other"
    slot: Optional[str] = None
    confidence: float = 1.0        # 0.0–1.0


class MemoryModel(BaseModel):
    id: str
    user_id: str
    memory: str
    type: str                      # "profile_fact" | "preference" | "episodic_event" | "temp_state" | "task_state" | "other"
    slot: Optional[str] = None
    status: str = "active"         # "active" | "archived" | "deleted"
    created_at: str
    valid_until: Optional[str] = None
    decay_half_life_days: Optional[int] = None
    confidence: float = 1.0
    supersedes: List[str] = []
    source_turn_id: Optional[str] = None
    extra: Dict = {}

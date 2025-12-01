from ..models import FactCandidate, MemoryModel
from ..storage.sqlite_store import SqliteStore


class TemporalEngine:
    """
    Responsible for:
    - Mapping FactCandidate -> MemoryModel (type, TTL, decay)
    - Conflict resolution (superseding old slot memories)
    - Temporal filtering + ranking on read.

    Day 1: only method signatures; no real logic.
    """

    def __init__(self, metadata_store: SqliteStore) -> None:
        self.metadata_store = metadata_store

    def process_write_batch(
        self,
        facts: list[FactCandidate],
        user_id: str,
        source_turn_id: str | None = None,
    ) -> list[MemoryModel]:
        """
        Turn a list of FactCandidate into MemoryModel objects.
        Day 1: return empty list.
        """
        return []

    def filter_and_rank(
        self,
        memories: list[MemoryModel],
    ) -> list[MemoryModel]:
        """
        Temporal filtering and ranking for reads.
        Day 1: return input as-is.
        """
        return memories

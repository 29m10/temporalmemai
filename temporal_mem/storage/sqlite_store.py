from ..models import MemoryModel


class SqliteStore:
    """
    SQLite-based metadata store.

    Day 1: just method signatures; implementation comes later.
    """

    def __init__(self, path: str = "~/.temporal_mem/history.db") -> None:
        self.path = path

    def insert(self, mem: MemoryModel) -> None:
        raise NotImplementedError

    def get_by_id(self, mem_id: str) -> MemoryModel | None:
        raise NotImplementedError

    def update_status(self, mem_id: str, new_status: str) -> None:
        raise NotImplementedError

    def get_active_by_slot(self, user_id: str, slot: str) -> list[MemoryModel]:
        raise NotImplementedError

    def list_by_user(self, user_id: str, status: str = "active") -> list[MemoryModel]:
        raise NotImplementedError

    def list_by_ids(self, ids: list[str]) -> list[MemoryModel]:
        raise NotImplementedError

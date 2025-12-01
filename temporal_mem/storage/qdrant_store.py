from typing import List, Optional, Dict, Any
from ..models import MemoryModel


class QdrantStore:
    """
    Qdrant-based vector store.

    Day 1: just method signatures; no actual Qdrant calls yet.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "temporal_mem_default",
        vector_size: int = 1536,
    ) -> None:
        self.host = host
        self.port = port
        self.collection = collection
        self.vector_size = vector_size

    def upsert(self, mem: MemoryModel, embedding: List[float]) -> None:
        raise NotImplementedError

    def delete(self, mem_id: str) -> None:
        raise NotImplementedError

    def search(
        self,
        user_id: str,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[str, float]]:
        raise NotImplementedError

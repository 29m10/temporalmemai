from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .llm.extractor import FactExtractor
from .temporal.engine import TemporalEngine
from .storage.sqlite_store import SqliteStore
from .storage.qdrant_store import QdrantStore
from .embedding.openai_embedder import OpenAIEmbedder
from .models import MemoryModel


class Memory:
    """
    Public facade. Day 1: wiring only, no real behavior.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}

        sqlite_path = config.get("sqlite_path", "~/.temporal_mem/history.db")
        qdrant_host = config.get("qdrant_host", "localhost")
        qdrant_port = int(config.get("qdrant_port", 6333))
        qdrant_collection = config.get("qdrant_collection", "temporal_mem_default")

        openai_api_key = config.get("openai_api_key")
        embed_model = config.get("embed_model", "text-embedding-3-small")
        llm_model = config.get("llm_model", "gpt-4.1-mini")
        llm_temp = float(config.get("llm_temperature", 0.0))

        self.metadata_store = SqliteStore(path=sqlite_path)
        self.temporal_engine = TemporalEngine(metadata_store=self.metadata_store)
        self.fact_extractor = FactExtractor(
            api_key=openai_api_key,
            model=llm_model,
            temperature=llm_temp,
        )
        self.embedder = OpenAIEmbedder(
            api_key=openai_api_key,
            model=embed_model,
        )
        self.vector_store = QdrantStore(
            host=qdrant_host,
            port=qdrant_port,
            collection=qdrant_collection,
            vector_size=1536,
        )

    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Day 1: return empty results.
        """
        return {"results": []}

    def search(
        self,
        query: str,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Day 1: return empty results.
        """
        return {"results": []}

    def list(
        self,
        user_id: str,
        status: str = "active",
    ) -> Dict[str, Any]:
        """
        Day 1: return empty results.
        """
        return {"results": []}

    def delete(self, memory_id: str) -> None:
        """
        Day 1: no-op.
        """
        return None

    def update(self, memory_id: str, new_content: str) -> Optional[Dict[str, Any]]:
        """
        Day 1: return None.
        """
        return None

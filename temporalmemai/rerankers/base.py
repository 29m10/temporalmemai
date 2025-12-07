# temporalmemai/rerankers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """Minimal interface for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        `documents` is a list of dicts (each representing a candidate).
        Must return the same docs, ordered, each with `rerank_score` float.
        """
        raise NotImplementedError

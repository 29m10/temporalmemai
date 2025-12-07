# temporalmemai/rerankers/factory.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import BaseReranker


# lazy imports to avoid hard deps for everything
def create_reranker(cfg: dict[str, Any] | None) -> BaseReranker | None:
    if not cfg:
        return None

    provider = cfg.get("provider")
    provider_cfg = cfg.get("config", {}) or {}

    if provider == "cohere":
        from .cohere_reranker import CohereReranker

        return CohereReranker(provider_cfg)

    if provider == "huggingface":
        from .huggingface_reranker import HuggingFaceReranker

        return HuggingFaceReranker(provider_cfg)

    if provider in ("llm", "llm_reranker", "openai"):
        from .llm_reranker import LLMReranker

        return LLMReranker(provider_cfg)

    # unknown / disabled
    return None

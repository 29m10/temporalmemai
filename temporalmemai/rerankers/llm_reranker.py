# temporalmemai/rerankers/llm_reranker.py

from __future__ import annotations

import os
import re
from typing import Any

from .base import BaseReranker

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


_DEFAULT_PROMPT = """You are a relevance scoring assistant. Given a query and a document, you need to score how relevant the document is to the query.

Score the relevance on a scale from 0.0 to 1.0, where:
- 1.0 = Perfectly relevant and directly answers the query
- 0.8-0.9 = Highly relevant with good information
- 0.6-0.7 = Moderately relevant with some useful information
- 0.4-0.5 = Slightly relevant with limited useful information
- 0.0-0.3 = Not relevant or no useful information

Query: "{query}"
Document: "{document}"

Provide only a single numerical score between 0.0 and 1.0. Do not include any explanation or additional text.
"""


class LLMReranker(BaseReranker):
    """
    LLM-based reranker.

    Config example:
    {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": null,              # optional, else env OPENAI_API_KEY
        "top_k": 5,
        "temperature": 0.0,
        "max_tokens": 50,
        "scoring_prompt": "...optional override..."
    }

    NOTE: This calls the LLM once per document (simple but slower).
    Good for small candidate sets (like 5-10 docs).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required for LLMReranker. Install with: pip install openai"
            )

        self.provider: str = config.get("provider", "openai")
        self.model: str = config.get("model", "gpt-4o-mini")
        self.temperature: float = float(config.get("temperature", 0.0))
        self.max_tokens: int = int(config.get("max_tokens", 50))
        self.top_k: int | None = config.get("top_k")

        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing for LLMReranker")

        # For now we only support OpenAI; you can extend this later by branching on self.provider
        if self.provider != "openai":
            raise ValueError(f"Unsupported LLM provider for LLMReranker: {self.provider}")

        self.client = OpenAI(api_key=api_key)

        self.scoring_prompt: str = config.get("scoring_prompt") or _DEFAULT_PROMPT

    # ------------------------ helpers ------------------------ #

    @staticmethod
    def _extract_score(text: str) -> float:
        """
        Extract a float between 0.0 and 1.0 from the LLM response.
        """
        pattern = r"\b([01](?:\.\d+)?)\b"
        matches = re.findall(pattern, text)
        if matches:
            try:
                val = float(matches[0])
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        # If parsing fails, return neutral-ish score
        return 0.5

    def _score_pair(self, query: str, document: str) -> float:
        """
        Call the LLM once to score (query, document).
        """
        prompt = self.scoring_prompt.format(query=query, document=document)

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        content = resp.choices[0].message.content or ""
        return self._extract_score(content.strip())

    # ----------------------- main API ------------------------ #

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank docs using LLM-based scoring.
        """
        if not documents:
            return documents

        scored_docs: list[dict[str, Any]] = []

        for doc in documents:
            # Extract doc text
            if "memory" in doc:
                text = doc["memory"]
            elif "text" in doc:
                text = doc["text"]
            elif "content" in doc:
                text = doc["content"]
            else:
                text = str(doc)

            try:
                score = self._score_pair(query, text)
            except Exception:
                score = 0.5  # neutral fallback

            d = doc.copy()
            d["rerank_score"] = float(score)
            scored_docs.append(d)

        # Sort by rerank_score desc
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Limit results
        k = top_k or self.top_k
        if k:
            scored_docs = scored_docs[:k]

        return scored_docs

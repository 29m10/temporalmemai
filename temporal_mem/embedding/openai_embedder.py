# temporal_mem/embedding/openai_embedder.py

from __future__ import annotations

import os

from openai import OpenAI


class OpenAIEmbedder:
    """
    Simple OpenAI embedding wrapper.

    Responsibilities:
    - Create a client
    - Embed a single text or list of texts
    - Return list[float] for single, list[list[float]] for batch
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbedder")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    @property
    def vector_size(self) -> int:
        """
        Return the vector dimension size for the current model.
        """
        # OpenAI embedding model dimensions
        if "text-embedding-3-large" in self.model:
            return 3072
        elif "text-embedding-3-small" in self.model:
            return 1536
        elif "text-embedding-ada-002" in self.model:
            return 1536
        else:
            # Default to 1536 for unknown models (most common)
            return 1536

    def embed_one(self, text: str) -> list[float]:
        """
        Embed a single text and return its embedding vector.
        """
        text = text or ""
        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return resp.data[0].embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts and return list of vectors.
        """
        if not texts:
            return []
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

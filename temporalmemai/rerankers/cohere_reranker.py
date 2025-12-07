# temporalmemai/rerankers/cohere_reranker.py
import os
from typing import Any

from .base import BaseReranker

try:
    import cohere

    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class CohereReranker(BaseReranker):
    def __init__(self, config: dict[str, Any]) -> None:
        if not COHERE_AVAILABLE:
            raise ImportError("cohere is required for CohereReranker. Install: pip install cohere")

        self.model: str = config.get("model", "rerank-english-v3.0")
        self.api_key: str = config.get("api_key") or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY missing for CohereReranker")

        self.top_k: int | None = config.get("top_k")
        self.return_documents: bool = bool(config.get("return_documents", False))
        self.max_chunks_per_doc: int | None = config.get("max_chunks_per_doc")

        self.client = cohere.Client(self.api_key)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not documents:
            return documents

        doc_texts = []
        for doc in documents:
            if "memory" in doc:
                doc_texts.append(doc["memory"])
            elif "text" in doc:
                doc_texts.append(doc["text"])
            elif "content" in doc:
                doc_texts.append(doc["content"])
            else:
                doc_texts.append(str(doc))

        try:
            k = top_k or self.top_k or len(documents)
            resp = self.client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=k,
                return_documents=self.return_documents,
                max_chunks_per_doc=self.max_chunks_per_doc,
            )

            reranked: list[dict[str, Any]] = []
            for item in resp.results:
                d = documents[item.index].copy()
                d["rerank_score"] = float(item.relevance_score)
                reranked.append(d)
            return reranked

        except Exception:
            # graceful fallback
            for d in documents:
                d["rerank_score"] = 0.0
            k = top_k or self.top_k
            return documents[:k] if k else documents

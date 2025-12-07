# temporalmemai/rerankers/huggingface_reranker.py

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseReranker

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class HuggingFaceReranker(BaseReranker):
    """
    HuggingFace Transformers-based reranker.

    Expects config like:
    {
        "model": "BAAI/bge-reranker-base",
        "device": "cuda" | "cpu" | None,
        "batch_size": 32,
        "max_length": 512,
        "top_k": 10,
        "normalize": True
    }
    """

    def __init__(self, config: dict[str, Any]) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for HuggingFaceReranker. "
                "Install with: pip install transformers torch"
            )

        self.model_name: str = config.get("model", "BAAI/bge-reranker-base")
        self.batch_size: int = int(config.get("batch_size", 32))
        self.max_length: int = int(config.get("max_length", 512))
        self.top_k: int | None = config.get("top_k")
        self.normalize: bool = bool(config.get("normalize", True))

        device_cfg = config.get("device")
        if device_cfg:
            self.device = device_cfg
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank documents using a cross-encoder.

        Each input doc is a dict; we'll look for:
        - "memory" (preferred)
        - "text"
        - "content"
        or fallback to str(doc).
        """
        if not documents:
            return documents

        # 1) Extract text for each doc
        texts: list[str] = []
        for doc in documents:
            if "memory" in doc:
                texts.append(doc["memory"])
            elif "text" in doc:
                texts.append(doc["text"])
            elif "content" in doc:
                texts.append(doc["content"])
            else:
                texts.append(str(doc))

        scores: list[float] = []

        # 2) Process in batches
        import torch  # safe here because TRANSFORMERS_AVAILABLE is True

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_pairs = [[query, t] for t in batch_texts]

            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1).cpu().numpy()

            batch_scores = [float(logits)] if logits.ndim == 0 else logits.tolist()

            scores.extend(batch_scores)

        # 3) Normalize scores if requested
        if self.normalize and len(scores) > 0:
            arr = np.array(scores, dtype=float)
            denom = float(arr.max() - arr.min() + 1e-8)
            arr = (arr - arr.min()) / denom
            scores = arr.tolist()

        # 4) Zip docs + scores and sort
        doc_score_pairs = list(zip(documents, scores, strict=True))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        final_top_k = top_k or self.top_k
        if final_top_k:
            doc_score_pairs = doc_score_pairs[:final_top_k]

        reranked: list[dict[str, Any]] = []
        for doc, score in doc_score_pairs:
            d = doc.copy()
            d["rerank_score"] = float(score)
            reranked.append(d)

        return reranked

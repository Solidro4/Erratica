from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from typing import Protocol


class SimilarityScorer(Protocol):
    name: str

    def score(self, reference: str, candidate: str) -> float:
        ...


@dataclass
class HybridStringSimilarityScorer:
    name: str = "hybrid_string"

    def score(self, reference: str, candidate: str) -> float:
        reference_normalized = self._normalize_text(reference)
        candidate_normalized = self._normalize_text(candidate)
        sequence_score = difflib.SequenceMatcher(
            None,
            reference_normalized,
            candidate_normalized,
        ).ratio()
        reference_tokens = self._extract_similarity_tokens(reference_normalized)
        if not reference_tokens:
            return sequence_score
        token_hits = sum(token in candidate_normalized for token in reference_tokens)
        token_score = token_hits / len(reference_tokens)
        return (sequence_score + token_score) / 2.0

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower().replace("–", "-").replace("â€“", "-")
        normalized = re.sub(r"\s+", " ", lowered)
        return normalized.strip()

    def _extract_similarity_tokens(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9\-]{4,}", text)
        seen: set[str] = set()
        filtered: list[str] = []
        for token in tokens:
            if token in {"that", "with", "will", "from", "your", "have", "once"}:
                continue
            if token in seen:
                continue
            seen.add(token)
            filtered.append(token)
        return filtered[:12]


class EmbeddingSimilarityScorer:
    name = "embedding_cosine"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it before using EmbeddingSimilarityScorer."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def score(self, reference: str, candidate: str) -> float:
        embeddings = self.model.encode([reference, candidate], normalize_embeddings=True)
        reference_vector = self._to_list(embeddings[0])
        candidate_vector = self._to_list(embeddings[1])
        cosine = self._cosine_similarity(reference_vector, candidate_vector)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _to_list(self, vector: object) -> list[float]:
        if hasattr(vector, "tolist"):
            return [float(value) for value in vector.tolist()]
        return [float(value) for value in vector]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)

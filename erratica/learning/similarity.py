from __future__ import annotations

import difflib
import json
import math
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol

_STOP_TOKENS = {"that", "with", "will", "from", "your", "have", "once", "this", "been", "they"}


class SimilarityScorer(Protocol):
    name: str

    def score(self, reference: str, candidate: str) -> float:
        ...


@dataclass
class HybridStringSimilarityScorer:
    """Dependency-free default: character alignment plus key-token coverage.

    It rewards answers that share wording with the reference, so it cannot
    tell that two differently phrased answers mean the same thing. Use an
    embedding scorer when paraphrases matter.
    """

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
        candidate_tokens = set(re.findall(r"[a-z0-9\-]+", candidate_normalized))
        token_hits = sum(token in candidate_tokens for token in reference_tokens)
        token_score = token_hits / len(reference_tokens)
        return (sequence_score + token_score) / 2.0

    def _normalize_text(self, text: str) -> str:
        lowered = text.lower()
        for dash in ("–", "—", "−"):
            lowered = lowered.replace(dash, "-")
        return re.sub(r"\s+", " ", lowered).strip()

    def _extract_similarity_tokens(self, text: str) -> list[str]:
        seen: set[str] = set()
        filtered: list[str] = []
        for token in re.findall(r"[a-z0-9\-]{4,}", text):
            if token in _STOP_TOKENS or token in seen:
                continue
            seen.add(token)
            filtered.append(token)
        return filtered[:24]


class _EmbeddingScorer:
    """Cosine similarity over cached embeddings. Subclasses provide ``_embed``."""

    name = "embedding_cosine"

    def __init__(self, cache_size: int = 4096) -> None:
        self._cache: dict[str, list[float]] = {}
        self._cache_size = cache_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = [float(value) for value in self._embed([text])[0]]
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[text] = vector
        return vector

    def score(self, reference: str, candidate: str) -> float:
        cosine = _cosine_similarity(self.embed(reference), self.embed(candidate))
        return max(0.0, min(1.0, cosine))


class EmbeddingSimilarityScorer(_EmbeddingScorer):
    """Semantic similarity with a local sentence-transformers model."""

    name = "embedding_cosine"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        super().__init__()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install erratica[embeddings]"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [list(vector.tolist() if hasattr(vector, "tolist") else vector) for vector in embeddings]


class OllamaEmbeddingScorer(_EmbeddingScorer):
    """Semantic similarity through a local Ollama embedding model.

    Needs no Python dependencies: it talks to the Ollama server over HTTP.
    Pull a model first, for example ``ollama pull nomic-embed-text``.
    """

    name = "ollama_embedding"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            self.host + "/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["embeddings"]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)

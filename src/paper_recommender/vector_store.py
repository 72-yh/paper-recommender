from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VectorSearchResult:
    vector_id: int
    score: float


class ExactVectorIndex:
    def __init__(self, vector_ids: np.ndarray, vectors: np.ndarray) -> None:
        self.vector_ids = np.asarray(vector_ids, dtype=np.int64)
        self.vectors = _normalize_matrix(vectors)

    @classmethod
    def from_items(cls, items: dict[int, np.ndarray]) -> ExactVectorIndex:
        vector_ids = np.array(list(items.keys()), dtype=np.int64)
        vectors = np.array(list(items.values()), dtype=np.float32)
        return cls(vector_ids, vectors)

    @classmethod
    def load(cls, path: str | Path) -> ExactVectorIndex:
        with np.load(path) as data:
            return cls(data["vector_ids"], data["vectors"])

    def save(self, path: str | Path) -> None:
        np.savez(path, vector_ids=self.vector_ids, vectors=self.vectors)

    def get(self, vector_id: int) -> np.ndarray | None:
        matches = np.where(self.vector_ids == vector_id)[0]
        if len(matches) == 0:
            return None
        return self.vectors[matches[0]].copy()

    def search(self, query: np.ndarray, top_k: int) -> list[VectorSearchResult]:
        if top_k <= 0 or len(self.vector_ids) == 0:
            return []

        normalized_query = _normalize_vector(query)
        scores = self.vectors @ normalized_query
        ordered_indices = np.argsort(scores)[::-1][:top_k]
        return [
            VectorSearchResult(vector_id=int(self.vector_ids[index]), score=float(scores[index]))
            for index in ordered_indices
        ]


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(normalized)
    if norm == 0:
        return normalized.copy()
    return normalized / norm


def _normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vectors, dtype=np.float32)
    if len(normalized) == 0:
        return normalized.copy()

    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    return np.divide(normalized, norms, out=np.zeros_like(normalized), where=norms != 0)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from paper_recommender.vector_store import ExactVectorIndex, VectorSearchResult

_INT8_SEARCH_CHUNK_SIZE = 65_536


class SearchableIndex(Protocol):
    def get(self, vector_id: int) -> np.ndarray | None: ...

    def search(self, query: np.ndarray, top_k: int) -> list[VectorSearchResult]: ...


@dataclass(frozen=True)
class RecallResult:
    queries: int
    k: int
    recall: float


class PcaFloatVectorIndex:
    def __init__(
        self,
        vector_ids: np.ndarray,
        vectors: np.ndarray,
        mean: np.ndarray,
        components: np.ndarray,
    ) -> None:
        self.vector_ids = np.asarray(vector_ids, dtype=np.int64)
        self.vectors = _normalize_matrix(vectors)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)

    @property
    def pca_dimensions(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def from_exact_index(
        cls,
        index: ExactVectorIndex,
        *,
        pca_dimensions: int,
    ) -> PcaFloatVectorIndex:
        original_dimensions = _original_dimensions(index.vectors)
        if len(index.vector_ids) == 0:
            return cls(
                np.array([], dtype=np.int64),
                np.zeros((0, pca_dimensions), dtype=np.float32),
                np.zeros(original_dimensions, dtype=np.float32),
                np.zeros((original_dimensions, pca_dimensions), dtype=np.float32),
            )
        mean, components, projected = _fit_pca(index.vectors, pca_dimensions)
        return cls(index.vector_ids.copy(), projected, mean, components)

    @classmethod
    def load(cls, path: str | Path) -> PcaFloatVectorIndex:
        with np.load(path) as data:
            return cls(
                data["vector_ids"],
                data["vectors"],
                data["mean"],
                data["components"],
            )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            vector_ids=self.vector_ids,
            vectors=self.vectors,
            mean=self.mean,
            components=self.components,
        )

    def get(self, vector_id: int) -> np.ndarray | None:
        matches = np.where(self.vector_ids == vector_id)[0]
        if len(matches) == 0:
            return None
        return self.vectors[matches[0]].copy()

    def search(self, query: np.ndarray, top_k: int) -> list[VectorSearchResult]:
        if top_k <= 0 or len(self.vector_ids) == 0:
            return []

        query_projected = self._project_query(query)
        scores = self.vectors @ query_projected
        ordered_indices = np.argsort(scores)[::-1][:top_k]
        return [
            VectorSearchResult(vector_id=int(self.vector_ids[index]), score=float(scores[index]))
            for index in ordered_indices
        ]

    def _project_query(self, query: np.ndarray) -> np.ndarray:
        vector = np.asarray(query, dtype=np.float32)
        projected = (vector - self.mean) @ self.components
        return _normalize_vector(projected)


class Int8VectorIndex:
    def __init__(self, vector_ids: np.ndarray, codes: np.ndarray, scales: np.ndarray) -> None:
        self.vector_ids = np.asarray(vector_ids, dtype=np.int64)
        self.codes = np.asarray(codes, dtype=np.int8)
        self.scales = np.asarray(scales, dtype=np.float32)
        self._row_norms = _decoded_int8_row_norms(self.codes, self.scales)

    @classmethod
    def from_exact_index(cls, index: ExactVectorIndex) -> Int8VectorIndex:
        if len(index.vector_ids) == 0:
            dimensions = _original_dimensions(index.vectors)
            return cls(
                np.array([], dtype=np.int64),
                np.zeros((0, dimensions), dtype=np.int8),
                np.ones(dimensions, dtype=np.float32),
            )
        vectors = np.asarray(index.vectors, dtype=np.float32)
        scales = _quantization_scales(vectors)
        codes = np.clip(np.rint(vectors / scales), -127, 127).astype(np.int8)
        return cls(index.vector_ids.copy(), codes, scales)

    @classmethod
    def load(cls, path: str | Path) -> Int8VectorIndex:
        with np.load(path) as data:
            return cls(data["vector_ids"], data["codes"], data["scales"])

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            vector_ids=self.vector_ids,
            codes=self.codes,
            scales=self.scales,
        )

    def get(self, vector_id: int) -> np.ndarray | None:
        matches = np.where(self.vector_ids == vector_id)[0]
        if len(matches) == 0:
            return None
        return _normalize_vector(self._decode_rows([int(matches[0])])[0])

    def search(self, query: np.ndarray, top_k: int) -> list[VectorSearchResult]:
        if top_k <= 0 or len(self.vector_ids) == 0:
            return []

        normalized_query = _normalize_vector(query)
        weighted_query = normalized_query * self.scales
        scores = _int8_cosine_scores(self.codes, weighted_query, self._row_norms)
        ordered_indices = np.argsort(scores)[::-1][:top_k]
        return [
            VectorSearchResult(vector_id=int(self.vector_ids[index]), score=float(scores[index]))
            for index in ordered_indices
        ]

    def _decode_rows(self, rows: list[int] | None) -> np.ndarray:
        codes = self.codes if rows is None else self.codes[rows]
        return np.asarray(codes, dtype=np.float32) * self.scales


class PcaInt8VectorIndex:
    def __init__(
        self,
        vector_ids: np.ndarray,
        codes: np.ndarray,
        mean: np.ndarray,
        components: np.ndarray,
        scales: np.ndarray,
    ) -> None:
        self.vector_ids = np.asarray(vector_ids, dtype=np.int64)
        self.codes = np.asarray(codes, dtype=np.int8)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)

    @property
    def pca_dimensions(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def from_exact_index(
        cls,
        index: ExactVectorIndex,
        *,
        pca_dimensions: int,
    ) -> PcaInt8VectorIndex:
        original_dimensions = _original_dimensions(index.vectors)
        if len(index.vector_ids) == 0:
            return cls(
                np.array([], dtype=np.int64),
                np.zeros((0, pca_dimensions), dtype=np.int8),
                np.zeros(original_dimensions, dtype=np.float32),
                np.zeros((original_dimensions, pca_dimensions), dtype=np.float32),
                np.ones(pca_dimensions, dtype=np.float32),
            )
        if pca_dimensions <= 0 or pca_dimensions > original_dimensions:
            raise ValueError("pca_dimensions must be between 1 and the original vector dimension")
        if pca_dimensions > len(index.vector_ids):
            raise ValueError("pca_dimensions cannot exceed the number of indexed vectors")

        mean, components, projected = _fit_pca(index.vectors, pca_dimensions)
        scales = _quantization_scales(projected)
        codes = np.clip(np.rint(projected / scales), -127, 127).astype(np.int8)
        return cls(index.vector_ids.copy(), codes, mean, components, scales)

    @classmethod
    def load(cls, path: str | Path) -> PcaInt8VectorIndex:
        with np.load(path) as data:
            return cls(
                data["vector_ids"],
                data["codes"],
                data["mean"],
                data["components"],
                data["scales"],
            )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            vector_ids=self.vector_ids,
            codes=self.codes,
            mean=self.mean,
            components=self.components,
            scales=self.scales,
        )

    def get(self, vector_id: int) -> np.ndarray | None:
        matches = np.where(self.vector_ids == vector_id)[0]
        if len(matches) == 0:
            return None
        return _normalize_vector(self._decode_rows([int(matches[0])])[0])

    def search(self, query: np.ndarray, top_k: int) -> list[VectorSearchResult]:
        if top_k <= 0 or len(self.vector_ids) == 0:
            return []

        decoded = _normalize_matrix(self._decode_rows(None))
        query_projected = self._project_query(query)
        scores = decoded @ query_projected
        ordered_indices = np.argsort(scores)[::-1][:top_k]
        return [
            VectorSearchResult(vector_id=int(self.vector_ids[index]), score=float(scores[index]))
            for index in ordered_indices
        ]

    def _project_query(self, query: np.ndarray) -> np.ndarray:
        vector = np.asarray(query, dtype=np.float32)
        projected = (vector - self.mean) @ self.components
        return _normalize_vector(projected)

    def _decode_rows(self, rows: list[int] | None) -> np.ndarray:
        codes = self.codes if rows is None else self.codes[rows]
        return np.asarray(codes, dtype=np.float32) * self.scales


def recall_at_k(
    baseline: ExactVectorIndex,
    candidate: SearchableIndex,
    *,
    query_vector_ids: list[int],
    k: int,
) -> RecallResult:
    if k <= 0 or not query_vector_ids:
        return RecallResult(queries=0, k=k, recall=0.0)

    recall_sum = 0.0
    queries = 0
    for vector_id in query_vector_ids:
        query = baseline.get(vector_id)
        if query is None:
            continue

        expected = {result.vector_id for result in baseline.search(query, k)}
        actual = {result.vector_id for result in candidate.search(query, k)}
        if not expected:
            continue
        recall_sum += len(expected & actual) / len(expected)
        queries += 1

    if queries == 0:
        return RecallResult(queries=0, k=k, recall=0.0)
    return RecallResult(queries=queries, k=k, recall=recall_sum / queries)


def _original_dimensions(vectors: np.ndarray) -> int:
    if vectors.ndim != 2:
        return 0
    return int(vectors.shape[1])


def _quantization_scales(vectors: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(vectors), axis=0)
    return np.where(max_abs == 0, 1.0, max_abs / 127.0).astype(np.float32)


def _decoded_int8_row_norms(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    norms = np.zeros(len(codes), dtype=np.float32)
    scale_squares = np.square(scales, dtype=np.float32)
    for start in range(0, len(codes), _INT8_SEARCH_CHUNK_SIZE):
        end = start + _INT8_SEARCH_CHUNK_SIZE
        chunk = np.asarray(codes[start:end], dtype=np.float32)
        norms[start:end] = np.sqrt(np.square(chunk) @ scale_squares)
    return norms


def _int8_cosine_scores(
    codes: np.ndarray,
    weighted_query: np.ndarray,
    row_norms: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(len(codes), dtype=np.float32)
    for start in range(0, len(codes), _INT8_SEARCH_CHUNK_SIZE):
        end = start + _INT8_SEARCH_CHUNK_SIZE
        raw_scores = np.asarray(codes[start:end], dtype=np.float32) @ weighted_query
        np.divide(
            raw_scores,
            row_norms[start:end],
            out=scores[start:end],
            where=row_norms[start:end] != 0,
        )
    return scores


def _fit_pca(vectors: np.ndarray, pca_dimensions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original_dimensions = _original_dimensions(vectors)
    if pca_dimensions <= 0 or pca_dimensions > original_dimensions:
        raise ValueError("pca_dimensions must be between 1 and the original vector dimension")
    if pca_dimensions > len(vectors):
        raise ValueError("pca_dimensions cannot exceed the number of indexed vectors")

    matrix = np.asarray(vectors, dtype=np.float32)
    mean = matrix.mean(axis=0).astype(np.float32)
    centered = matrix - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:pca_dimensions].T.astype(np.float32)
    projected = centered @ components
    return mean, components, projected


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

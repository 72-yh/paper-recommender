from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from paper_recommender.vector_store import (
    ExactVectorIndex,
    VectorSearchResult,
    candidate_row_indices,
    top_k_indices,
)

_INT8_SEARCH_CHUNK_SIZE = 65_536
_RECALL_BATCH_SIZE = 32


class SearchableIndex(Protocol):
    def get(self, vector_id: int) -> np.ndarray | None: ...

    def search(self, query: np.ndarray, top_k: int) -> list[VectorSearchResult]: ...


@dataclass(frozen=True)
class RecallResult:
    queries: int
    k: int
    recall: float


@dataclass(frozen=True)
class _ClusteredInt8Arrays:
    vector_ids: np.ndarray
    codes: np.ndarray
    row_norms: np.ndarray
    offsets: np.ndarray


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
        ordered_indices = top_k_indices(scores, top_k)
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

    def save_mmap(self, path: str | Path) -> None:
        _save_int8_mmap_arrays(
            path,
            vector_ids=self.vector_ids,
            codes=self.codes,
            scales=self.scales,
            row_norms=self._row_norms,
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
        ordered_indices = top_k_indices(scores, top_k)
        return [
            VectorSearchResult(vector_id=int(self.vector_ids[index]), score=float(scores[index]))
            for index in ordered_indices
        ]

    def search_subset(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_vector_ids: list[int] | tuple[int, ...] | np.ndarray,
    ) -> list[VectorSearchResult]:
        return _int8_search_subset(
            self.vector_ids,
            self.codes,
            self.scales,
            self._row_norms,
            query,
            top_k,
            candidate_vector_ids,
        )

    def _decode_rows(self, rows: list[int] | None) -> np.ndarray:
        codes = self.codes if rows is None else self.codes[rows]
        return np.asarray(codes, dtype=np.float32) * self.scales


class MmapInt8VectorIndex:
    def __init__(
        self,
        vector_ids: np.ndarray,
        codes: np.ndarray,
        scales: np.ndarray,
        row_norms: np.ndarray,
    ) -> None:
        self.vector_ids = _coerce_array(vector_ids, np.int64)
        self.codes = _coerce_array(codes, np.int8)
        self.scales = _coerce_array(scales, np.float32)
        self._row_norms = _coerce_array(row_norms, np.float32)

    @classmethod
    def load(cls, path: str | Path) -> MmapInt8VectorIndex:
        directory = Path(path)
        return cls(
            np.load(directory / "vector_ids.npy", mmap_mode="r"),
            np.load(directory / "codes.npy", mmap_mode="r"),
            np.load(directory / "scales.npy", mmap_mode="r"),
            np.load(directory / "row_norms.npy", mmap_mode="r"),
        )

    def save(self, path: str | Path) -> None:
        _save_int8_mmap_arrays(
            path,
            vector_ids=self.vector_ids,
            codes=self.codes,
            scales=self.scales,
            row_norms=self._row_norms,
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
        ordered_indices = top_k_indices(scores, top_k)
        return [
            VectorSearchResult(vector_id=int(self.vector_ids[index]), score=float(scores[index]))
            for index in ordered_indices
        ]

    def search_subset(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_vector_ids: list[int] | tuple[int, ...] | np.ndarray,
    ) -> list[VectorSearchResult]:
        return _int8_search_subset(
            self.vector_ids,
            self.codes,
            self.scales,
            self._row_norms,
            query,
            top_k,
            candidate_vector_ids,
        )

    def _decode_rows(self, rows: list[int] | None) -> np.ndarray:
        codes = self.codes if rows is None else self.codes[rows]
        return np.asarray(codes, dtype=np.float32) * self.scales


class IvfInt8VectorIndex:
    def __init__(
        self,
        base_index: MmapInt8VectorIndex,
        cluster_ids: np.ndarray,
        centroids: np.ndarray,
        *,
        clustered_arrays: _ClusteredInt8Arrays | None = None,
        nprobe: int = 32,
        min_candidate_multiplier: int = 200,
    ) -> None:
        self.base_index = base_index
        self.vector_ids = base_index.vector_ids
        self.cluster_ids = _coerce_array(cluster_ids, np.uint16)
        self.centroids = _normalize_matrix(centroids)
        if len(self.cluster_ids) != len(self.vector_ids):
            raise ValueError("cluster_ids length must match vector_ids length")
        if self.centroids.ndim != 2 or self.centroids.shape[1] != self.base_index.codes.shape[1]:
            raise ValueError("centroids must match vector dimensions")
        if np.any(self.cluster_ids >= len(self.centroids)):
            raise ValueError("cluster_ids contain values outside centroid range")
        self.clustered_arrays = self._validate_clustered_arrays(clustered_arrays)
        self.nprobe = max(1, min(int(nprobe), len(self.centroids)))
        self.min_candidate_multiplier = max(1, int(min_candidate_multiplier))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        nprobe: int = 32,
        min_candidate_multiplier: int = 200,
    ) -> IvfInt8VectorIndex:
        directory = Path(path)
        return cls(
            MmapInt8VectorIndex.load(directory),
            np.load(directory / "cluster_ids.npy", mmap_mode="r"),
            np.load(directory / "centroids.npy", mmap_mode="r"),
            clustered_arrays=_load_clustered_int8_arrays(directory),
            nprobe=nprobe,
            min_candidate_multiplier=min_candidate_multiplier,
        )

    def get(self, vector_id: int) -> np.ndarray | None:
        return self.base_index.get(vector_id)

    def search(self, query: np.ndarray, top_k: int) -> list[VectorSearchResult]:
        if top_k <= 0 or len(self.vector_ids) == 0:
            return []

        if self.clustered_arrays is not None:
            clusters = self._candidate_clusters(query, top_k)
            return _int8_search_cluster_slices(
                self.clustered_arrays.vector_ids,
                self.clustered_arrays.codes,
                self.base_index.scales,
                self.clustered_arrays.row_norms,
                self.clustered_arrays.offsets,
                query,
                top_k,
                clusters,
            )

        row_indices = self._candidate_rows(query, top_k)
        return _int8_search_rows(
            self.vector_ids,
            self.base_index.codes,
            self.base_index.scales,
            self.base_index._row_norms,
            query,
            top_k,
            row_indices,
        )

    def search_subset(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_vector_ids: list[int] | tuple[int, ...] | np.ndarray,
    ) -> list[VectorSearchResult]:
        if top_k <= 0 or len(self.vector_ids) == 0:
            return []

        candidate_rows = candidate_row_indices(self.vector_ids, candidate_vector_ids)
        if len(candidate_rows) == 0:
            return []

        if self.clustered_arrays is not None:
            clusters = self._candidate_clusters(query, top_k, candidate_rows=candidate_rows)
            return _int8_search_cluster_slices(
                self.clustered_arrays.vector_ids,
                self.clustered_arrays.codes,
                self.base_index.scales,
                self.clustered_arrays.row_norms,
                self.clustered_arrays.offsets,
                query,
                top_k,
                clusters,
                candidate_vector_ids=self.vector_ids[candidate_rows],
            )

        row_indices = self._candidate_rows(query, top_k, candidate_rows=candidate_rows)
        return _int8_search_rows(
            self.vector_ids,
            self.base_index.codes,
            self.base_index.scales,
            self.base_index._row_norms,
            query,
            top_k,
            row_indices,
        )

    def _candidate_rows(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_rows: np.ndarray | None = None,
    ) -> np.ndarray:
        selected_clusters = self._candidate_clusters(query, top_k, candidate_rows=candidate_rows)
        source_rows = (
            np.arange(len(self.vector_ids), dtype=np.int64)
            if candidate_rows is None
            else np.asarray(candidate_rows, dtype=np.int64)
        )
        return source_rows[np.isin(self.cluster_ids[source_rows], selected_clusters)]

    def _candidate_clusters(
        self,
        query: np.ndarray,
        top_k: int,
        candidate_rows: np.ndarray | None = None,
    ) -> np.ndarray:
        cluster_order = self._cluster_order(query)
        source_rows = (
            np.arange(len(self.vector_ids), dtype=np.int64)
            if candidate_rows is None
            else np.asarray(candidate_rows, dtype=np.int64)
        )
        if len(source_rows) == 0:
            return source_rows

        min_candidates = min(
            len(source_rows),
            max(top_k, top_k * self.min_candidate_multiplier),
        )
        probe_count = self.nprobe
        while True:
            selected_clusters = cluster_order[:probe_count]
            if candidate_rows is None and self.clustered_arrays is not None:
                candidate_count = int(
                    np.sum(
                        self.clustered_arrays.offsets[selected_clusters + 1]
                        - self.clustered_arrays.offsets[selected_clusters]
                    )
                )
            else:
                candidate_count = int(
                    np.count_nonzero(np.isin(self.cluster_ids[source_rows], selected_clusters))
                )
            if candidate_count >= min_candidates or probe_count == len(cluster_order):
                return selected_clusters
            probe_count = min(len(cluster_order), probe_count * 2)

    def _cluster_order(self, query: np.ndarray) -> np.ndarray:
        normalized_query = _normalize_vector(query)
        scores = self.centroids @ normalized_query
        return top_k_indices(scores, len(scores))

    def _validate_clustered_arrays(
        self,
        arrays: _ClusteredInt8Arrays | None,
    ) -> _ClusteredInt8Arrays | None:
        if arrays is None:
            return None
        vector_ids = _coerce_array(arrays.vector_ids, np.int64)
        codes = _coerce_array(arrays.codes, np.int8)
        row_norms = _coerce_array(arrays.row_norms, np.float32)
        offsets = _coerce_array(arrays.offsets, np.int64)
        if len(vector_ids) != len(self.vector_ids):
            raise ValueError("clustered_vector_ids length must match vector_ids length")
        if codes.shape != self.base_index.codes.shape:
            raise ValueError("clustered_codes shape must match base codes shape")
        if len(row_norms) != len(self.vector_ids):
            raise ValueError("clustered_row_norms length must match vector_ids length")
        if len(offsets) != len(self.centroids) + 1:
            raise ValueError("cluster_offsets length must equal n_clusters + 1")
        if offsets[0] != 0 or offsets[-1] != len(self.vector_ids):
            raise ValueError("cluster_offsets must span every clustered row")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("cluster_offsets must be non-decreasing")
        return _ClusteredInt8Arrays(vector_ids, codes, row_norms, offsets)


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
        ordered_indices = top_k_indices(scores, top_k)
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

    query_vectors = _baseline_query_vectors(baseline, query_vector_ids)
    if len(query_vectors) == 0:
        return RecallResult(queries=0, k=k, recall=0.0)

    expected_sets = _exact_top_k_sets(baseline, query_vectors, k)
    if isinstance(candidate, Int8VectorIndex):
        actual_sets = _int8_top_k_sets(candidate, query_vectors, k)
    else:
        actual_sets = [
            {result.vector_id for result in candidate.search(query, k)}
            for query in query_vectors
        ]

    recall_sum = 0.0
    queries = 0
    for expected, actual in zip(expected_sets, actual_sets, strict=True):
        if not expected:
            continue
        recall_sum += len(expected & actual) / len(expected)
        queries += 1

    if queries == 0:
        return RecallResult(queries=0, k=k, recall=0.0)
    return RecallResult(queries=queries, k=k, recall=recall_sum / queries)


def _baseline_query_vectors(
    baseline: ExactVectorIndex,
    query_vector_ids: list[int],
) -> np.ndarray:
    baseline_positions = {
        int(vector_id): index for index, vector_id in enumerate(baseline.vector_ids)
    }
    queries: list[np.ndarray] = []
    for vector_id in query_vector_ids:
        position = baseline_positions.get(vector_id)
        if position is None:
            continue
        queries.append(baseline.vectors[position])
    return np.array(queries, dtype=np.float32)


def _exact_top_k_sets(
    index: ExactVectorIndex,
    queries: np.ndarray,
    k: int,
) -> list[set[int]]:
    normalized_queries = _normalize_matrix(queries)
    results: list[set[int]] = []
    for start in range(0, len(normalized_queries), _RECALL_BATCH_SIZE):
        query_batch = normalized_queries[start : start + _RECALL_BATCH_SIZE]
        scores = query_batch @ index.vectors.T
        results.extend(_score_rows_to_vector_id_sets(scores, index.vector_ids, k))
    return results


def _int8_top_k_sets(
    index: Int8VectorIndex,
    queries: np.ndarray,
    k: int,
) -> list[set[int]]:
    normalized_queries = _normalize_matrix(queries)
    results: list[set[int]] = []
    for start in range(0, len(normalized_queries), _RECALL_BATCH_SIZE):
        query_batch = normalized_queries[start : start + _RECALL_BATCH_SIZE]
        weighted_queries = query_batch * index.scales
        scores = np.zeros((len(query_batch), len(index.vector_ids)), dtype=np.float32)
        for chunk_start in range(0, len(index.codes), _INT8_SEARCH_CHUNK_SIZE):
            chunk_end = chunk_start + _INT8_SEARCH_CHUNK_SIZE
            raw_scores = np.asarray(index.codes[chunk_start:chunk_end], dtype=np.float32) @ (
                weighted_queries.T
            )
            np.divide(
                raw_scores,
                index._row_norms[chunk_start:chunk_end, np.newaxis],
                out=raw_scores,
                where=index._row_norms[chunk_start:chunk_end, np.newaxis] != 0,
            )
            scores[:, chunk_start:chunk_end] = raw_scores.T
        results.extend(_score_rows_to_vector_id_sets(scores, index.vector_ids, k))
    return results


def _score_rows_to_vector_id_sets(
    scores: np.ndarray,
    vector_ids: np.ndarray,
    k: int,
) -> list[set[int]]:
    return [
        {int(vector_ids[index]) for index in top_k_indices(score_row, k)}
        for score_row in scores
    ]


def _original_dimensions(vectors: np.ndarray) -> int:
    if vectors.ndim != 2:
        return 0
    return int(vectors.shape[1])


def _quantization_scales(vectors: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(vectors), axis=0)
    return np.where(max_abs == 0, 1.0, max_abs / 127.0).astype(np.float32)


def _coerce_array(array: np.ndarray, dtype) -> np.ndarray:
    if array.dtype == dtype:
        return array
    return np.asarray(array, dtype=dtype)


def _save_int8_mmap_arrays(
    path: str | Path,
    *,
    vector_ids: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    row_norms: np.ndarray,
) -> None:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vector_ids.npy", np.asarray(vector_ids, dtype=np.int64))
    np.save(directory / "codes.npy", np.asarray(codes, dtype=np.int8))
    np.save(directory / "scales.npy", np.asarray(scales, dtype=np.float32))
    np.save(directory / "row_norms.npy", np.asarray(row_norms, dtype=np.float32))


def _load_clustered_int8_arrays(directory: Path) -> _ClusteredInt8Arrays | None:
    paths = {
        "vector_ids": directory / "clustered_vector_ids.npy",
        "codes": directory / "clustered_codes.npy",
        "row_norms": directory / "clustered_row_norms.npy",
        "offsets": directory / "cluster_offsets.npy",
    }
    existing = [path.exists() for path in paths.values()]
    if not any(existing):
        return None
    if not all(existing):
        missing = ", ".join(name for name, path in paths.items() if not path.exists())
        raise ValueError(f"incomplete clustered IVF arrays: missing {missing}")
    return _ClusteredInt8Arrays(
        np.load(paths["vector_ids"], mmap_mode="r"),
        np.load(paths["codes"], mmap_mode="r"),
        np.load(paths["row_norms"], mmap_mode="r"),
        np.load(paths["offsets"], mmap_mode="r"),
    )


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


def _int8_search_subset(
    vector_ids: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    row_norms: np.ndarray,
    query: np.ndarray,
    top_k: int,
    candidate_vector_ids: list[int] | tuple[int, ...] | np.ndarray,
) -> list[VectorSearchResult]:
    if top_k <= 0 or len(vector_ids) == 0:
        return []

    row_indices = candidate_row_indices(vector_ids, candidate_vector_ids)
    if len(row_indices) == 0:
        return []

    return _int8_search_rows(vector_ids, codes, scales, row_norms, query, top_k, row_indices)


def _int8_search_rows(
    vector_ids: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    row_norms: np.ndarray,
    query: np.ndarray,
    top_k: int,
    row_indices: np.ndarray,
) -> list[VectorSearchResult]:
    if top_k <= 0 or len(row_indices) == 0:
        return []

    normalized_query = _normalize_vector(query)
    weighted_query = normalized_query * scales
    selected_codes = codes[row_indices]
    selected_norms = row_norms[row_indices]
    scores = _int8_cosine_scores(selected_codes, weighted_query, selected_norms)
    ordered_indices = top_k_indices(scores, top_k)
    return [
        VectorSearchResult(
            vector_id=int(vector_ids[row_indices[index]]),
            score=float(scores[index]),
        )
        for index in ordered_indices
    ]


def _int8_search_cluster_slices(
    vector_ids: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    row_norms: np.ndarray,
    cluster_offsets: np.ndarray,
    query: np.ndarray,
    top_k: int,
    clusters: np.ndarray,
    *,
    candidate_vector_ids: np.ndarray | None = None,
) -> list[VectorSearchResult]:
    if top_k <= 0 or len(clusters) == 0:
        return []

    normalized_query = _normalize_vector(query)
    weighted_query = normalized_query * scales
    candidate_ids = (
        None if candidate_vector_ids is None else np.asarray(candidate_vector_ids, dtype=np.int64)
    )
    score_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    for cluster_id in clusters:
        start = int(cluster_offsets[int(cluster_id)])
        end = int(cluster_offsets[int(cluster_id) + 1])
        if start == end:
            continue

        cluster_vector_ids = vector_ids[start:end]
        cluster_codes = codes[start:end]
        cluster_norms = row_norms[start:end]
        if candidate_ids is not None:
            mask = np.isin(cluster_vector_ids, candidate_ids)
            if not np.any(mask):
                continue
            cluster_vector_ids = cluster_vector_ids[mask]
            cluster_codes = cluster_codes[mask]
            cluster_norms = cluster_norms[mask]

        score_parts.append(_int8_cosine_scores(cluster_codes, weighted_query, cluster_norms))
        id_parts.append(np.asarray(cluster_vector_ids, dtype=np.int64))

    if not score_parts:
        return []

    scores = np.concatenate(score_parts)
    result_vector_ids = np.concatenate(id_parts)
    ordered_indices = top_k_indices(scores, top_k)
    return [
        VectorSearchResult(
            vector_id=int(result_vector_ids[index]),
            score=float(scores[index]),
        )
        for index in ordered_indices
    ]


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

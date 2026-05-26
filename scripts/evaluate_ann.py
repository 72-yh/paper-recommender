from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paper_recommender.compressed_vector_store import (
    Int8VectorIndex,
    MmapInt8VectorIndex,
    RecallResult,
)
from paper_recommender.vector_store import ExactVectorIndex


class AnnEvaluationError(Exception):
    """Raised when an ANN candidate cannot be evaluated."""


@dataclass(frozen=True)
class AnnEvaluationReport:
    method: str
    input_path: Path
    input_kind: str
    output_path: Path
    indexed_vectors: int
    dimensions: int
    top_k: int
    sample_size: int
    build_seconds: float
    load_seconds: float
    search_p50_ms: float
    search_p95_ms: float
    search_max_ms: float
    output_bytes: int
    recall: RecallResult


def evaluate_ann(
    *,
    input_path: str | Path,
    input_kind: str,
    output_path: str | Path,
    method: str,
    top_k: int,
    sample_size: int,
    max_vectors: int | None = None,
    usearch_dtype: str = "f16",
    usearch_index_factory: Callable[..., object] | None = None,
) -> AnnEvaluationReport:
    if method != "usearch":
        raise AnnEvaluationError(f"Unsupported ANN method: {method}")
    if top_k <= 0:
        raise AnnEvaluationError("top_k must be positive")
    if sample_size <= 0:
        raise AnnEvaluationError("sample_size must be positive")

    input_path = Path(input_path)
    output_path = Path(output_path)
    vector_ids, vectors = _load_source_vectors(input_path, input_kind=input_kind, max_vectors=max_vectors)
    if len(vector_ids) == 0:
        raise AnnEvaluationError("input index does not contain vectors")

    exact_subset = ExactVectorIndex(vector_ids, vectors)
    query_vector_ids = vector_ids[: min(sample_size, len(vector_ids))].astype(int).tolist()
    query_vectors = exact_subset.vectors[: len(query_vector_ids)]

    factory = usearch_index_factory or _load_usearch_index_factory()
    start = time.perf_counter()
    ann = factory(ndim=int(vectors.shape[1]), metric="cos", dtype=usearch_dtype)
    ann.add(vector_ids.astype(np.uint64), vectors.astype(np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ann.save(output_path)
    build_seconds = time.perf_counter() - start

    start = time.perf_counter()
    loaded_ann = factory(ndim=int(vectors.shape[1]), metric="cos", dtype=usearch_dtype)
    loaded_ann.load(output_path)
    load_seconds = time.perf_counter() - start

    recall, search_latencies = _measure_ann_recall(
        exact_subset,
        loaded_ann,
        query_vectors=query_vectors,
        top_k=top_k,
    )

    return AnnEvaluationReport(
        method=method,
        input_path=input_path,
        input_kind=input_kind,
        output_path=output_path,
        indexed_vectors=len(vector_ids),
        dimensions=int(vectors.shape[1]),
        top_k=top_k,
        sample_size=len(query_vector_ids),
        build_seconds=build_seconds,
        load_seconds=load_seconds,
        search_p50_ms=_percentile(search_latencies, 50),
        search_p95_ms=_percentile(search_latencies, 95),
        search_max_ms=max(search_latencies, default=0.0),
        output_bytes=output_path.stat().st_size,
        recall=recall,
    )


def _load_source_vectors(
    input_path: Path,
    *,
    input_kind: str,
    max_vectors: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if input_kind == "exact":
        index = ExactVectorIndex.load(input_path)
        return _limit(index.vector_ids, index.vectors, max_vectors=max_vectors)
    if input_kind == "int8":
        index = Int8VectorIndex.load(input_path)
        vector_ids, codes = _limit(index.vector_ids, index.codes, max_vectors=max_vectors)
        return vector_ids, _normalize_matrix(np.asarray(codes, dtype=np.float32) * index.scales)
    if input_kind == "int8_mmap":
        index = MmapInt8VectorIndex.load(input_path)
        vector_ids, codes = _limit(index.vector_ids, index.codes, max_vectors=max_vectors)
        return vector_ids, _normalize_matrix(np.asarray(codes, dtype=np.float32) * index.scales)
    raise AnnEvaluationError(f"Unsupported input kind: {input_kind}")


def _limit(
    vector_ids: np.ndarray,
    vectors: np.ndarray,
    *,
    max_vectors: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_vectors is None:
        return np.asarray(vector_ids, dtype=np.int64), np.asarray(vectors, dtype=np.float32)
    if max_vectors <= 0:
        raise AnnEvaluationError("max_vectors must be positive")
    return (
        np.asarray(vector_ids[:max_vectors], dtype=np.int64),
        np.asarray(vectors[:max_vectors], dtype=np.float32),
    )


def _load_usearch_index_factory():
    try:
        from usearch.index import Index
    except ModuleNotFoundError as exc:
        raise AnnEvaluationError(
            "USearch is not installed. Install the optional ANN dependency with "
            "`pip install .[ann]` before running method=usearch."
        ) from exc
    return Index


def _measure_ann_recall(
    exact: ExactVectorIndex,
    ann,
    *,
    query_vectors: np.ndarray,
    top_k: int,
) -> tuple[RecallResult, list[float]]:
    recall_sum = 0.0
    measured_queries = 0
    latencies: list[float] = []
    for query in query_vectors:
        expected = {result.vector_id for result in exact.search(query, top_k)}
        start = time.perf_counter()
        matches = ann.search(query.astype(np.float32), count=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        actual = {int(key) for key in np.asarray(matches.keys).ravel()[:top_k]}
        if not expected:
            continue
        recall_sum += len(expected & actual) / len(expected)
        measured_queries += 1

    if measured_queries == 0:
        return RecallResult(queries=0, k=top_k, recall=0.0), latencies
    return (
        RecallResult(queries=measured_queries, k=top_k, recall=recall_sum / measured_queries),
        latencies,
    )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    if percentile == 50:
        return float(statistics.median(ordered))
    rank = round((percentile / 100) * (len(ordered) - 1))
    return float(ordered[rank])


def _normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if len(matrix) == 0:
        return matrix.copy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate an optional ANN serving index."
    )
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--input-kind", choices=("exact", "int8", "int8_mmap"), required=True)
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument("--method", choices=("usearch",), default="usearch")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--max-vectors", type=int)
    parser.add_argument("--usearch-dtype", default="f16")
    args = parser.parse_args()

    report = evaluate_ann(
        input_path=args.input_path,
        input_kind=args.input_kind,
        output_path=args.output_path,
        method=args.method,
        top_k=args.top_k,
        sample_size=args.sample_size,
        max_vectors=args.max_vectors,
        usearch_dtype=args.usearch_dtype,
    )
    print(_format_report(report))


def _format_report(report: AnnEvaluationReport) -> str:
    return (
        "ANN report: "
        f"method={report.method} "
        f"input_kind={report.input_kind} "
        f"indexed_vectors={report.indexed_vectors} "
        f"dimensions={report.dimensions} "
        f"build_seconds={report.build_seconds:.4f} "
        f"load_seconds={report.load_seconds:.4f} "
        f"search_p50_ms={report.search_p50_ms:.3f} "
        f"search_p95_ms={report.search_p95_ms:.3f} "
        f"search_max_ms={report.search_max_ms:.3f} "
        f"output_bytes={report.output_bytes} "
        f"recall@{report.recall.k}={report.recall.recall:.4f} "
        f"queries={report.recall.queries}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from paper_recommender.compressed_vector_store import IvfInt8VectorIndex, MmapInt8VectorIndex


@dataclass(frozen=True)
class IvfEvaluationReport:
    index_path: Path
    indexed_vectors: int
    top_k: int
    queries: int
    nprobe: int
    recall: float
    exact_p50_ms: float
    exact_p95_ms: float
    ivf_p50_ms: float
    ivf_p95_ms: float


def evaluate_ivf_int8_index(
    *,
    index_path: str | Path,
    top_k: int = 10,
    sample_size: int = 100,
    nprobe: int = 32,
    seed: int = 13,
) -> IvfEvaluationReport:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    index_path = Path(index_path)
    exact = MmapInt8VectorIndex.load(index_path)
    candidate = IvfInt8VectorIndex.load(index_path, nprobe=nprobe)
    query_rows = _sample_rows(len(exact.vector_ids), sample_size, seed)

    exact_latencies: list[float] = []
    ivf_latencies: list[float] = []
    recall_sum = 0.0
    queries = 0
    for row in query_rows:
        query = exact.get(int(exact.vector_ids[row]))
        if query is None:
            continue

        start = time.perf_counter()
        expected = {result.vector_id for result in exact.search(query, top_k)}
        exact_latencies.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        actual = {result.vector_id for result in candidate.search(query, top_k)}
        ivf_latencies.append((time.perf_counter() - start) * 1000)

        if expected:
            recall_sum += len(expected & actual) / len(expected)
            queries += 1

    recall = 0.0 if queries == 0 else recall_sum / queries
    return IvfEvaluationReport(
        index_path=index_path,
        indexed_vectors=len(exact.vector_ids),
        top_k=top_k,
        queries=queries,
        nprobe=nprobe,
        recall=recall,
        exact_p50_ms=_percentile(exact_latencies, 50),
        exact_p95_ms=_percentile(exact_latencies, 95),
        ivf_p50_ms=_percentile(ivf_latencies, 50),
        ivf_p95_ms=_percentile(ivf_latencies, 95),
    )


def _sample_rows(total_rows: int, sample_size: int, seed: int) -> np.ndarray:
    limit = min(total_rows, sample_size)
    if limit == total_rows:
        return np.arange(total_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total_rows, size=limit, replace=False).astype(np.int64))


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate IVF int8 recall against exact int8_mmap search."
    )
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    report = evaluate_ivf_int8_index(
        index_path=args.index_path,
        top_k=args.top_k,
        sample_size=args.sample_size,
        nprobe=args.nprobe,
        seed=args.seed,
    )
    print(_format_report(report))


def _format_report(report: IvfEvaluationReport) -> str:
    return (
        "IVF evaluation: "
        f"index_path={report.index_path} "
        f"indexed_vectors={report.indexed_vectors} "
        f"top_k={report.top_k} "
        f"queries={report.queries} "
        f"nprobe={report.nprobe} "
        f"recall@{report.top_k}={report.recall:.4f} "
        f"exact_p50_ms={report.exact_p50_ms:.3f} "
        f"exact_p95_ms={report.exact_p95_ms:.3f} "
        f"ivf_p50_ms={report.ivf_p50_ms:.3f} "
        f"ivf_p95_ms={report.ivf_p95_ms:.3f}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from paper_recommender.compressed_vector_store import (
    Int8VectorIndex,
    IvfInt8VectorIndex,
    MmapInt8VectorIndex,
)
from paper_recommender.recommender import recommend
from paper_recommender.storage import connect_db
from paper_recommender.vector_store import ExactVectorIndex


@dataclass(frozen=True)
class LatencyStats:
    name: str
    query_count: int
    p50_ms: float
    p95_ms: float
    max_ms: float
    total_results: int


@dataclass(frozen=True)
class ServingBenchmarkSummary:
    db_path: Path
    index_path: Path
    index_kind: str
    indexed_papers: int
    load_seconds: float
    unfiltered: LatencyStats
    filtered: LatencyStats | None


def benchmark_serving_index(
    *,
    db_path: str | Path,
    index_path: str | Path,
    index_kind: str,
    query_count: int = 20,
    top_k: int = 10,
    categories: tuple[str, ...] = (),
) -> ServingBenchmarkSummary:
    db_path = Path(db_path)
    index_path = Path(index_path)
    start = time.perf_counter()
    index = _load_index(index_path, index_kind)
    load_seconds = time.perf_counter() - start

    conn = connect_db(db_path)
    try:
        indexed_papers = _count_indexed_papers(conn)
        query_arxiv_ids = sample_query_arxiv_ids(conn, limit=query_count)
        unfiltered = _measure_recommendations(
            conn,
            index,
            query_arxiv_ids=query_arxiv_ids,
            top_k=top_k,
            categories=(),
            name="unfiltered",
        )
        filtered = None
        if categories:
            filtered = _measure_recommendations(
                conn,
                index,
                query_arxiv_ids=query_arxiv_ids,
                top_k=top_k,
                categories=categories,
                name="filtered",
            )
    finally:
        conn.close()

    return ServingBenchmarkSummary(
        db_path=db_path,
        index_path=index_path,
        index_kind=index_kind,
        indexed_papers=indexed_papers,
        load_seconds=load_seconds,
        unfiltered=unfiltered,
        filtered=filtered,
    )


def sample_query_arxiv_ids(conn, *, limit: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT arxiv_id
        FROM papers
        WHERE active = 1 AND vector_id IS NOT NULL
        ORDER BY vector_id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [row["arxiv_id"] for row in rows]


def _load_index(index_path: Path, index_kind: str):
    if index_kind == "exact":
        return ExactVectorIndex.load(index_path)
    if index_kind == "int8":
        return Int8VectorIndex.load(index_path)
    if index_kind == "int8_mmap":
        return MmapInt8VectorIndex.load(index_path)
    if index_kind == "ivf_int8_mmap":
        return IvfInt8VectorIndex.load(index_path)
    raise ValueError(f"Unsupported index kind: {index_kind}")


def _count_indexed_papers(conn) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS value
        FROM papers
        WHERE active = 1 AND vector_id IS NOT NULL
        """
    ).fetchone()
    return int(row["value"])


def _measure_recommendations(
    conn,
    index,
    *,
    query_arxiv_ids: list[str],
    top_k: int,
    categories: tuple[str, ...],
    name: str,
) -> LatencyStats:
    elapsed_ms: list[float] = []
    total_results = 0
    for arxiv_id in query_arxiv_ids:
        start = time.perf_counter()
        results = recommend(conn, index, arxiv_id, top_k=top_k, categories=categories)
        elapsed_ms.append((time.perf_counter() - start) * 1000)
        total_results += len(results)

    return LatencyStats(
        name=name,
        query_count=len(elapsed_ms),
        p50_ms=_percentile(elapsed_ms, 50),
        p95_ms=_percentile(elapsed_ms, 95),
        max_ms=max(elapsed_ms, default=0.0),
        total_results=total_results,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Paper Recommender serving latency.")
    parser.add_argument("--db-path", type=Path, default=Path("data/paper_recommender_1m.db"))
    parser.add_argument("--index-path", type=Path, default=Path("data/vectors_1m_int8_mmap"))
    parser.add_argument(
        "--index-kind",
        choices=("exact", "int8", "int8_mmap", "ivf_int8_mmap"),
        default="int8_mmap",
    )
    parser.add_argument("--query-count", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--category", action="append", default=[])
    args = parser.parse_args()

    summary = benchmark_serving_index(
        db_path=args.db_path,
        index_path=args.index_path,
        index_kind=args.index_kind,
        query_count=args.query_count,
        top_k=args.top_k,
        categories=tuple(args.category),
    )
    print(_format_summary(summary))


def _format_summary(summary: ServingBenchmarkSummary) -> str:
    parts = [
        "Serving benchmark:",
        f"db_path={summary.db_path}",
        f"index_path={summary.index_path}",
        f"index_kind={summary.index_kind}",
        f"indexed_papers={summary.indexed_papers}",
        f"load_seconds={summary.load_seconds:.4f}",
        _format_latency(summary.unfiltered),
    ]
    if summary.filtered is not None:
        parts.append(_format_latency(summary.filtered))
    return " ".join(parts)


def _format_latency(stats: LatencyStats) -> str:
    return (
        f"{stats.name}_queries={stats.query_count} "
        f"{stats.name}_p50_ms={stats.p50_ms:.3f} "
        f"{stats.name}_p95_ms={stats.p95_ms:.3f} "
        f"{stats.name}_max_ms={stats.max_ms:.3f} "
        f"{stats.name}_total_results={stats.total_results}"
    )


if __name__ == "__main__":
    main()

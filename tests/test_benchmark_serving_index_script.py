import subprocess
import sys

import numpy as np

from paper_recommender.models import Paper
from paper_recommender.storage import connect_db, init_db, upsert_paper
from paper_recommender.vector_store import ExactVectorIndex
from scripts.benchmark_serving_index import benchmark_serving_index, sample_query_arxiv_ids


def test_benchmark_serving_index_measures_recommendation_latency(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    _write_db(db_path)
    _exact_index().save(index_path)

    summary = benchmark_serving_index(
        db_path=db_path,
        index_path=index_path,
        index_kind="exact",
        query_count=2,
        top_k=2,
    )

    assert summary.index_kind == "exact"
    assert summary.indexed_papers == 3
    assert summary.load_seconds >= 0
    assert summary.unfiltered.query_count == 2
    assert summary.unfiltered.p50_ms >= 0
    assert summary.unfiltered.max_ms >= summary.unfiltered.p50_ms


def test_benchmark_serving_index_can_measure_filtered_recommendations(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    _write_db(db_path)
    _exact_index().save(index_path)

    summary = benchmark_serving_index(
        db_path=db_path,
        index_path=index_path,
        index_kind="exact",
        query_count=2,
        top_k=2,
        categories=("cs.CL",),
    )

    assert summary.filtered is not None
    assert summary.filtered.name == "filtered"
    assert summary.filtered.query_count == 2


def test_sample_query_arxiv_ids_uses_active_indexed_papers(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    _write_db(db_path)
    conn = connect_db(db_path)
    try:
        assert sample_query_arxiv_ids(conn, limit=2) == ["0000.00001", "0000.00002"]
    finally:
        conn.close()


def test_benchmark_serving_index_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_serving_index.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Benchmark Paper Recommender serving latency" in result.stdout


def _write_db(path) -> None:
    conn = connect_db(path)
    init_db(conn)
    try:
        for vector_id, category in [(1, "cs.CL"), (2, "cs.CL"), (3, "math.OC")]:
            upsert_paper(
                conn,
                Paper(
                    arxiv_id=f"0000.0000{vector_id}",
                    vector_id=vector_id,
                    active=True,
                    oai_datestamp="2024-01-03",
                    published_date="2024-01-01",
                    updated_date=None,
                    primary_category=category,
                    categories=(category,),
                    content_hash=f"hash-{vector_id}",
                ),
            )
    finally:
        conn.close()


def _exact_index() -> ExactVectorIndex:
    return ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    )

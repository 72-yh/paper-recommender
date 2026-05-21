import subprocess
import sys

import numpy as np
import pytest

from paper_recommender.compressed_vector_store import Int8VectorIndex
from paper_recommender.models import Paper
from paper_recommender.storage import connect_db, init_db, set_pipeline_state, upsert_paper
from paper_recommender.vector_store import ExactVectorIndex
from scripts.preflight_artifacts import ArtifactPreflightError, preflight_artifacts


def test_preflight_artifacts_accepts_matching_int8_db_and_index(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors_int8.npz"
    exact = _exact_index()
    _write_db(db_path, vector_ids=(1, 2, 3))
    Int8VectorIndex.from_exact_index(exact).save(index_path)

    summary = preflight_artifacts(
        db_path=db_path,
        index_path=index_path,
        index_kind="int8",
        min_indexed_papers=3,
    )

    assert summary.indexed_papers == 3
    assert summary.index_vectors == 3
    assert summary.dimensions == 3
    assert summary.last_oai_datestamp == "2024-01-03"


def test_preflight_artifacts_accepts_matching_exact_db_and_index(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    exact = _exact_index()
    _write_db(db_path, vector_ids=(1, 2, 3))
    exact.save(index_path)

    summary = preflight_artifacts(
        db_path=db_path,
        index_path=index_path,
        index_kind="exact",
    )

    assert summary.index_kind == "exact"
    assert summary.index_vectors == 3


def test_preflight_artifacts_rejects_missing_files(tmp_path) -> None:
    with pytest.raises(ArtifactPreflightError, match="DB file does not exist"):
        preflight_artifacts(
            db_path=tmp_path / "missing.db",
            index_path=tmp_path / "vectors.npz",
            index_kind="int8",
        )


def test_preflight_artifacts_rejects_wrong_index_kind(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    _write_db(db_path, vector_ids=(1, 2, 3))
    _exact_index().save(index_path)

    with pytest.raises(ArtifactPreflightError, match="missing required arrays"):
        preflight_artifacts(
            db_path=db_path,
            index_path=index_path,
            index_kind="int8",
        )


def test_preflight_artifacts_rejects_count_mismatch(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors_int8.npz"
    _write_db(db_path, vector_ids=(1, 2, 3))
    Int8VectorIndex.from_exact_index(
        ExactVectorIndex.from_items({1: np.array([1.0, 0.0, 0.0], dtype=np.float32)})
    ).save(index_path)

    with pytest.raises(ArtifactPreflightError, match="count mismatch"):
        preflight_artifacts(
            db_path=db_path,
            index_path=index_path,
            index_kind="int8",
        )


def test_preflight_artifacts_rejects_vector_id_mismatch(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors_int8.npz"
    _write_db(db_path, vector_ids=(1, 2, 3))
    mismatched = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    )
    Int8VectorIndex.from_exact_index(mismatched).save(index_path)

    with pytest.raises(ArtifactPreflightError, match="Vector IDs do not match"):
        preflight_artifacts(
            db_path=db_path,
            index_path=index_path,
            index_kind="int8",
        )


def test_preflight_artifacts_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/preflight_artifacts.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Validate Paper Recommender deployment artifacts" in result.stdout


def _exact_index() -> ExactVectorIndex:
    return ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            3: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    )


def _write_db(path, *, vector_ids: tuple[int, ...]) -> None:
    conn = connect_db(path)
    init_db(conn)
    try:
        for vector_id in vector_ids:
            upsert_paper(
                conn,
                Paper(
                    arxiv_id=f"0000.0000{vector_id}",
                    vector_id=vector_id,
                    active=True,
                    oai_datestamp="2024-01-03",
                    published_date="2024-01-01",
                    updated_date=None,
                    primary_category="cs.CL",
                    categories=("cs.CL",),
                    content_hash=f"hash-{vector_id}",
                ),
            )
        set_pipeline_state(conn, "last_successful_oai_datestamp", "2024-01-03")
    finally:
        conn.close()

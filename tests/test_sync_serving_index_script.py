from pathlib import Path
import subprocess
import sys

import numpy as np

from paper_recommender.compressed_vector_store import MmapInt8VectorIndex
from paper_recommender.compressed_vector_store import RecallResult
from paper_recommender.index_builder import IndexBuildSummary
from paper_recommender.vector_store import ExactVectorIndex
from scripts.evaluate_compression import CompressionReport
from scripts.sync_serving_index import sync_serving_index


def _summary(*, embedded: int = 0, deleted: int = 0) -> IndexBuildSummary:
    return IndexBuildSummary(
        batches_seen=1,
        records_seen=3,
        inserted=embedded,
        updated=0,
        unchanged=3 - embedded - deleted,
        deleted=deleted,
        embedded=embedded,
        checkpoints_written=1,
        last_datestamp="2024-01-02",
    )


def _report(input_path: Path, output_path: Path) -> CompressionReport:
    return CompressionReport(
        method="int8",
        input_path=input_path,
        output_path=output_path,
        pca_dimensions=None,
        sample_size=10,
        input_bytes=100,
        output_bytes=25,
        recall=RecallResult(queries=10, k=10, recall=0.99),
    )


def test_sync_serving_index_rebuilds_when_embeddings_change(tmp_path) -> None:
    calls: list[str] = []
    exact_path = tmp_path / "vectors.npz"
    serving_path = tmp_path / "vectors_int8.npz"

    def update_index(**_kwargs):
        calls.append("update")
        return _summary(embedded=2)

    def evaluate(**kwargs):
        calls.append("evaluate")
        assert kwargs["input_path"] == exact_path
        assert kwargs["output_path"] == serving_path
        assert kwargs["method"] == "int8"
        return _report(exact_path, serving_path)

    summary = sync_serving_index(
        db_path=tmp_path / "papers.db",
        exact_index_path=exact_path,
        serving_index_path=serving_path,
        update_index=update_index,
        evaluate=evaluate,
        record_report=False,
    )

    assert calls == ["update", "evaluate"]
    assert summary.rebuilt_serving_index is True
    assert summary.compression is not None


def test_sync_serving_index_skips_rebuild_when_no_vector_changes(tmp_path) -> None:
    def update_index(**_kwargs):
        return _summary()

    def evaluate(**_kwargs):
        raise AssertionError("serving index should not rebuild without vector changes")

    summary = sync_serving_index(
        db_path=tmp_path / "papers.db",
        exact_index_path=tmp_path / "vectors.npz",
        serving_index_path=tmp_path / "vectors_int8.npz",
        update_index=update_index,
        evaluate=evaluate,
        record_report=False,
    )

    assert summary.rebuilt_serving_index is False
    assert summary.compression is None


def test_sync_serving_index_force_rebuilds_without_vector_changes(tmp_path) -> None:
    calls: list[str] = []

    def update_index(**_kwargs):
        calls.append("update")
        return _summary()

    def evaluate(**kwargs):
        calls.append("evaluate")
        return _report(kwargs["input_path"], kwargs["output_path"])

    summary = sync_serving_index(
        db_path=tmp_path / "papers.db",
        exact_index_path=tmp_path / "vectors.npz",
        serving_index_path=tmp_path / "vectors_int8.npz",
        update_index=update_index,
        evaluate=evaluate,
        force_rebuild=True,
        record_report=False,
    )

    assert calls == ["update", "evaluate"]
    assert summary.rebuilt_serving_index is True


def test_sync_serving_index_passes_target_vector_count_to_update(tmp_path) -> None:
    seen_target_count: list[int | None] = []

    def update_index(**kwargs):
        seen_target_count.append(kwargs["target_vector_count"])
        return _summary()

    summary = sync_serving_index(
        db_path=tmp_path / "papers.db",
        exact_index_path=tmp_path / "vectors.npz",
        serving_index_path=tmp_path / "vectors_int8.npz",
        update_index=update_index,
        target_vector_count=1_001_000,
        record_report=False,
    )

    assert seen_target_count == [1_001_000]
    assert summary.rebuilt_serving_index is False


def test_sync_serving_index_can_rebuild_int8_mmap_serving_artifact(tmp_path) -> None:
    exact_path = tmp_path / "vectors.npz"
    serving_path = tmp_path / "vectors_int8_mmap"
    exact = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
        }
    )
    exact.save(exact_path)

    def update_index(**_kwargs):
        return _summary(embedded=1)

    summary = sync_serving_index(
        db_path=tmp_path / "papers.db",
        exact_index_path=exact_path,
        serving_index_path=serving_path,
        serving_index_kind="int8_mmap",
        update_index=update_index,
        record_report=False,
        sample_size=1,
    )
    loaded = MmapInt8VectorIndex.load(serving_path)

    assert summary.rebuilt_serving_index is True
    assert summary.compression is not None
    assert summary.compression.output_path == serving_path
    assert summary.compression.output_bytes > 0
    assert [result.vector_id for result in loaded.search(exact.get(1), 2)] == [1, 2]


def test_sync_serving_index_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/sync_serving_index.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Resume OAI updates" in result.stdout
    assert "--serving-index-kind" in result.stdout
    assert "--target-vector-count" in result.stdout

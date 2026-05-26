from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from paper_recommender.compressed_vector_store import Int8VectorIndex
from paper_recommender.vector_store import ExactVectorIndex
from scripts.evaluate_ann import AnnEvaluationError, evaluate_ann


def test_evaluate_ann_builds_usearch_candidate_and_measures_recall(tmp_path) -> None:
    input_path = tmp_path / "vectors.npz"
    output_path = tmp_path / "ann.usearch"
    _exact_index().save(input_path)

    report = evaluate_ann(
        input_path=input_path,
        input_kind="exact",
        output_path=output_path,
        method="usearch",
        top_k=2,
        sample_size=2,
        usearch_index_factory=FakeUsearchIndex,
    )

    assert report.method == "usearch"
    assert report.indexed_vectors == 4
    assert report.recall.k == 2
    assert report.recall.queries == 2
    assert report.recall.recall == 1.0
    assert report.build_seconds >= 0
    assert report.load_seconds >= 0
    assert report.search_p50_ms >= 0
    assert report.output_bytes == output_path.stat().st_size


def test_evaluate_ann_can_read_int8_source_vectors(tmp_path) -> None:
    input_path = tmp_path / "vectors_int8.npz"
    output_path = tmp_path / "ann.usearch"
    Int8VectorIndex.from_exact_index(_exact_index()).save(input_path)

    report = evaluate_ann(
        input_path=input_path,
        input_kind="int8",
        output_path=output_path,
        method="usearch",
        top_k=2,
        sample_size=2,
        usearch_index_factory=FakeUsearchIndex,
    )

    assert report.indexed_vectors == 4
    assert report.recall.recall >= 0.5


def test_evaluate_ann_rejects_unknown_method(tmp_path) -> None:
    input_path = tmp_path / "vectors.npz"
    _exact_index().save(input_path)

    with pytest.raises(AnnEvaluationError, match="Unsupported ANN method"):
        evaluate_ann(
            input_path=input_path,
            input_kind="exact",
            output_path=tmp_path / "ann.index",
            method="unknown",
            top_k=2,
            sample_size=2,
            usearch_index_factory=FakeUsearchIndex,
        )


def test_evaluate_ann_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_ann.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Build and evaluate an optional ANN serving index" in result.stdout


class FakeMatches:
    def __init__(self, keys: np.ndarray, distances: np.ndarray) -> None:
        self.keys = keys
        self.distances = distances


class FakeUsearchIndex:
    _saved: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __init__(self, *, ndim: int, metric: str, dtype: str) -> None:
        self.ndim = ndim
        self.metric = metric
        self.dtype = dtype
        self.keys: np.ndarray | None = None
        self.vectors: np.ndarray | None = None

    def add(self, keys, vectors, **_kwargs):
        self.keys = np.asarray(keys, dtype=np.uint64)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        return self.keys

    def search(self, vectors, count: int, **_kwargs):
        if self.keys is None or self.vectors is None:
            raise AssertionError("index must be built before search")
        query = np.asarray(vectors, dtype=np.float32)
        scores = self.vectors @ query
        ordered = np.argsort(scores)[::-1][:count]
        return FakeMatches(
            keys=self.keys[ordered],
            distances=(1.0 - scores[ordered]).astype(np.float32),
        )

    def save(self, path):
        if self.keys is None or self.vectors is None:
            raise AssertionError("index must be built before save")
        self._saved[str(path)] = (self.keys.copy(), self.vectors.copy())
        Path(path).write_bytes(b"fake-usearch-index")

    def load(self, path):
        self.keys, self.vectors = self._saved[str(path)]


def _exact_index() -> ExactVectorIndex:
    return ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    )

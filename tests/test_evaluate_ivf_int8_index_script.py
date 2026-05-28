import subprocess
import sys

import numpy as np

from paper_recommender.compressed_vector_store import Int8VectorIndex
from paper_recommender.vector_store import ExactVectorIndex
from scripts.evaluate_ivf_int8_index import evaluate_ivf_int8_index


def test_evaluate_ivf_int8_index_reports_recall_and_latency(tmp_path) -> None:
    index_path = tmp_path / "vectors_ivf_int8_mmap"
    exact = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.0, 1.0, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.9, 0.1], dtype=np.float32),
        }
    )
    Int8VectorIndex.from_exact_index(exact).save_mmap(index_path)
    np.save(index_path / "cluster_ids.npy", np.array([0, 0, 1, 1], dtype=np.uint16))
    np.save(
        index_path / "centroids.npy",
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )

    report = evaluate_ivf_int8_index(index_path=index_path, top_k=2, sample_size=2, nprobe=1)

    assert report.queries == 2
    assert report.recall >= 0.5
    assert report.ivf_p50_ms >= 0
    assert report.exact_p50_ms >= 0


def test_evaluate_ivf_int8_index_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/evaluate_ivf_int8_index.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Evaluate IVF int8 recall" in result.stdout

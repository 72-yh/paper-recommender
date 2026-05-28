import subprocess
import sys

import numpy as np

from paper_recommender.compressed_vector_store import Int8VectorIndex, IvfInt8VectorIndex
from paper_recommender.vector_store import ExactVectorIndex
from scripts.build_ivf_int8_index import build_ivf_int8_index


def test_build_ivf_int8_index_writes_searchable_cluster_files(tmp_path) -> None:
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

    report = build_ivf_int8_index(
        index_path=index_path,
        n_clusters=2,
        train_sample_size=4,
        iterations=2,
        assignment_batch_size=2,
        seed=7,
    )

    assert report.indexed_vectors == 4
    assert report.n_clusters == 2
    assert (index_path / "centroids.npy").exists()
    assert (index_path / "cluster_ids.npy").exists()
    assert (index_path / "cluster_offsets.npy").exists()
    assert (index_path / "clustered_vector_ids.npy").exists()
    assert (index_path / "clustered_codes.npy").exists()
    assert (index_path / "clustered_row_norms.npy").exists()
    loaded = IvfInt8VectorIndex.load(index_path, nprobe=1, min_candidate_multiplier=1)
    assert loaded.search(exact.get(1), top_k=1)


def test_build_ivf_int8_index_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_ivf_int8_index.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Build IVF cluster files" in result.stdout

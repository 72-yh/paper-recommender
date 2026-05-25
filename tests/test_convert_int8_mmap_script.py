import subprocess
import sys

import numpy as np

from paper_recommender.compressed_vector_store import Int8VectorIndex, MmapInt8VectorIndex
from paper_recommender.vector_store import ExactVectorIndex
from scripts.convert_int8_mmap import convert_int8_index_to_mmap


def test_convert_int8_index_to_mmap_writes_searchable_directory(tmp_path) -> None:
    input_path = tmp_path / "vectors_int8.npz"
    output_path = tmp_path / "vectors_int8_mmap"
    exact = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
        }
    )
    Int8VectorIndex.from_exact_index(exact).save(input_path)

    summary = convert_int8_index_to_mmap(input_path=input_path, output_path=output_path)
    loaded = MmapInt8VectorIndex.load(output_path)

    assert summary.input_path == input_path
    assert summary.output_path == output_path
    assert summary.vectors == 2
    assert summary.dimensions == 2
    assert summary.output_bytes > 0
    assert [result.vector_id for result in loaded.search(exact.get(1), 2)] == [1, 2]


def test_convert_int8_mmap_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/convert_int8_mmap.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Convert a compressed int8 NPZ index to mmap NPY files" in result.stdout

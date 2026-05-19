from pathlib import Path

import numpy as np

from paper_recommender.vector_store import ExactVectorIndex
from scripts.evaluate_compression import append_report, evaluate_compression


def test_evaluate_compression_writes_artifact_and_reports_recall(tmp_path) -> None:
    input_path = tmp_path / "vectors.npz"
    output_path = tmp_path / "compressed_vectors.npz"
    ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.1, 0.9, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    ).save(input_path)

    report = evaluate_compression(
        input_path=input_path,
        output_path=output_path,
        method="pca-int8",
        pca_dimensions=2,
        top_k=2,
        sample_size=3,
    )

    assert output_path.exists()
    assert report.method == "pca-int8"
    assert report.input_bytes == Path(input_path).stat().st_size
    assert report.output_bytes == Path(output_path).stat().st_size
    assert report.pca_dimensions == 2
    assert report.sample_size == 3
    assert report.recall.queries == 3
    assert report.recall.k == 2
    assert 0.0 <= report.recall.recall <= 1.0


def test_append_report_writes_jsonl_and_markdown(tmp_path) -> None:
    input_path = tmp_path / "vectors.npz"
    output_path = tmp_path / "compressed_vectors.npz"
    jsonl_path = tmp_path / "compression-runs.jsonl"
    markdown_path = tmp_path / "compression-runs.md"
    ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.1, 0.9, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    ).save(input_path)
    report = evaluate_compression(
        input_path=input_path,
        output_path=output_path,
        method="pca-int8",
        pca_dimensions=2,
        top_k=2,
        sample_size=3,
    )

    append_report(
        report,
        jsonl_path=jsonl_path,
        markdown_path=markdown_path,
        label="unit-test",
    )

    jsonl = jsonl_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert '"label": "unit-test"' in jsonl
    assert '"method": "pca-int8"' in jsonl
    assert '"pca_dimensions": 2' in jsonl
    assert "# Compression Evaluation Runs" in markdown
    assert "| unit-test |" in markdown


def test_evaluate_compression_supports_pca_float_and_int8(tmp_path) -> None:
    input_path = tmp_path / "vectors.npz"
    ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.1, 0.9, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    ).save(input_path)

    pca_report = evaluate_compression(
        input_path=input_path,
        output_path=tmp_path / "pca_float.npz",
        method="pca-float",
        pca_dimensions=2,
        top_k=2,
        sample_size=4,
    )
    int8_report = evaluate_compression(
        input_path=input_path,
        output_path=tmp_path / "int8.npz",
        method="int8",
        pca_dimensions=None,
        top_k=2,
        sample_size=4,
    )

    assert pca_report.method == "pca-float"
    assert pca_report.pca_dimensions == 2
    assert int8_report.method == "int8"
    assert int8_report.pca_dimensions is None

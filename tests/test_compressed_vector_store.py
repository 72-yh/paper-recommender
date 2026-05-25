import numpy as np
import pytest

from paper_recommender.compressed_vector_store import (
    Int8VectorIndex,
    MmapInt8VectorIndex,
    PcaFloatVectorIndex,
    PcaInt8VectorIndex,
    recall_at_k,
)
from paper_recommender.vector_store import ExactVectorIndex


def _baseline_index() -> ExactVectorIndex:
    return ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.1, 0.9, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    )


def test_pca_int8_index_search_matches_baseline_neighbors() -> None:
    exact = _baseline_index()
    compressed = PcaInt8VectorIndex.from_exact_index(exact, pca_dimensions=2)

    exact_results = exact.search(exact.get(1), top_k=3)
    compressed_results = compressed.search(exact.get(1), top_k=3)

    assert [result.vector_id for result in compressed_results][:2] == [
        result.vector_id for result in exact_results
    ][:2]
    assert compressed.codes.dtype == np.int8
    assert compressed.components.shape == (3, 2)


def test_pca_int8_index_save_and_load(tmp_path) -> None:
    path = tmp_path / "compressed_vectors.npz"
    compressed = PcaInt8VectorIndex.from_exact_index(_baseline_index(), pca_dimensions=2)

    compressed.save(path)
    loaded = PcaInt8VectorIndex.load(path)

    assert loaded.pca_dimensions == 2
    assert np.array_equal(loaded.vector_ids, compressed.vector_ids)
    assert np.array_equal(loaded.codes, compressed.codes)
    assert [result.vector_id for result in loaded.search(np.array([1.0, 0.0, 0.0]), 2)] == [1, 2]


def test_pca_float_index_save_load_and_search(tmp_path) -> None:
    path = tmp_path / "pca_float_vectors.npz"
    exact = _baseline_index()
    index = PcaFloatVectorIndex.from_exact_index(exact, pca_dimensions=2)

    index.save(path)
    loaded = PcaFloatVectorIndex.load(path)

    assert loaded.pca_dimensions == 2
    assert loaded.vectors.dtype == np.float32
    assert loaded.vectors.shape == (4, 2)
    assert [result.vector_id for result in loaded.search(exact.get(1), 2)] == [1, 2]


def test_int8_index_save_load_and_search(tmp_path) -> None:
    path = tmp_path / "int8_vectors.npz"
    exact = _baseline_index()
    index = Int8VectorIndex.from_exact_index(exact)

    index.save(path)
    loaded = Int8VectorIndex.load(path)

    assert loaded.codes.dtype == np.int8
    assert loaded.codes.shape == exact.vectors.shape
    assert [result.vector_id for result in loaded.search(exact.get(1), 2)] == [1, 2]


def test_int8_index_search_does_not_decode_all_rows_per_query(monkeypatch) -> None:
    exact = _baseline_index()
    index = Int8VectorIndex.from_exact_index(exact)
    original_decode_rows = index._decode_rows

    def fail_full_decode(rows):
        if rows is None:
            raise AssertionError("search should use the int8 score path")
        return original_decode_rows(rows)

    monkeypatch.setattr(index, "_decode_rows", fail_full_decode)

    assert [result.vector_id for result in index.search(exact.get(1), 2)] == [1, 2]


def test_int8_mmap_index_save_load_and_search(tmp_path) -> None:
    path = tmp_path / "int8_mmap"
    exact = _baseline_index()
    index = Int8VectorIndex.from_exact_index(exact)

    index.save_mmap(path)
    loaded = MmapInt8VectorIndex.load(path)

    assert isinstance(loaded.codes, np.memmap)
    assert isinstance(loaded.vector_ids, np.memmap)
    assert isinstance(loaded._row_norms, np.memmap)
    assert (path / "row_norms.npy").exists()
    assert [result.vector_id for result in loaded.search(exact.get(1), 2)] == [1, 2]


def test_int8_mmap_load_uses_saved_row_norms(monkeypatch, tmp_path) -> None:
    path = tmp_path / "int8_mmap"
    index = Int8VectorIndex.from_exact_index(_baseline_index())
    index.save_mmap(path)

    def fail_recompute(*_args, **_kwargs):
        raise AssertionError("mmap load should use saved row norms")

    monkeypatch.setattr(
        "paper_recommender.compressed_vector_store._decoded_int8_row_norms",
        fail_recompute,
    )

    loaded = MmapInt8VectorIndex.load(path)

    assert np.array_equal(loaded._row_norms, index._row_norms)


def test_pca_int8_rejects_invalid_dimensions() -> None:
    exact = _baseline_index()

    with pytest.raises(ValueError, match="pca_dimensions"):
        PcaInt8VectorIndex.from_exact_index(exact, pca_dimensions=0)

    with pytest.raises(ValueError, match="pca_dimensions"):
        PcaInt8VectorIndex.from_exact_index(exact, pca_dimensions=5)

    with pytest.raises(ValueError, match="pca_dimensions"):
        PcaFloatVectorIndex.from_exact_index(exact, pca_dimensions=5)


def test_pca_int8_empty_index_searches_safely() -> None:
    exact = ExactVectorIndex.from_items({})
    compressed = PcaInt8VectorIndex.from_exact_index(exact, pca_dimensions=2)
    pca_float = PcaFloatVectorIndex.from_exact_index(exact, pca_dimensions=2)
    int8 = Int8VectorIndex.from_exact_index(exact)

    assert compressed.search(np.array([1.0, 0.0]), top_k=5) == []
    assert compressed.get(1) is None
    assert pca_float.search(np.array([1.0, 0.0]), top_k=5) == []
    assert pca_float.get(1) is None
    assert int8.search(np.array([1.0, 0.0]), top_k=5) == []
    assert int8.get(1) is None


def test_recall_at_k_compares_compressed_results_to_exact_results() -> None:
    exact = _baseline_index()
    compressed = PcaInt8VectorIndex.from_exact_index(exact, pca_dimensions=2)

    result = recall_at_k(exact, compressed, query_vector_ids=[1, 3], k=2)

    assert result.queries == 2
    assert result.k == 2
    assert 0.5 <= result.recall <= 1.0


def test_recall_at_k_uses_direct_baseline_vector_lookup(monkeypatch) -> None:
    exact = _baseline_index()
    compressed = Int8VectorIndex.from_exact_index(exact)

    def fail_get(*_args, **_kwargs):
        raise AssertionError("recall should not linearly scan baseline ids for each query")

    monkeypatch.setattr(exact, "get", fail_get)

    result = recall_at_k(exact, compressed, query_vector_ids=[1, 3], k=2)

    assert result.queries == 2


def test_recall_at_k_batches_int8_candidate_search(monkeypatch) -> None:
    exact = _baseline_index()
    compressed = Int8VectorIndex.from_exact_index(exact)

    def fail_search(*_args, **_kwargs):
        raise AssertionError("recall should batch int8 candidate search")

    monkeypatch.setattr(compressed, "search", fail_search)

    result = recall_at_k(exact, compressed, query_vector_ids=[1, 3], k=2)

    assert result.queries == 2

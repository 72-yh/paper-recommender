import numpy as np
import pytest

from paper_recommender.vector_store import ExactVectorIndex, top_k_indices


def test_exact_vector_search_orders_by_cosine_similarity() -> None:
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.8, 0.2], dtype=np.float32),
            3: np.array([0.0, 1.0], dtype=np.float32),
        }
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)

    assert [item.vector_id for item in results] == [1, 2]
    assert results[0].score > results[1].score


def test_vector_index_save_and_load(tmp_path) -> None:
    path = tmp_path / "vectors.npz"
    index = ExactVectorIndex.from_items(
        {
            7: np.array([3.0, 4.0], dtype=np.float32),
            9: np.array([0.0, 5.0], dtype=np.float32),
        }
    )

    index.save(path)
    loaded = ExactVectorIndex.load(path)

    assert np.allclose(loaded.get(7), np.array([0.6, 0.8], dtype=np.float32))
    assert np.allclose(loaded.get(9), np.array([0.0, 1.0], dtype=np.float32))
    assert loaded.get(1) is None


def test_empty_vector_index_searches_safely() -> None:
    index = ExactVectorIndex.from_items({})

    assert index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3) == []
    assert index.get(1) is None


def test_search_returns_empty_for_non_positive_top_k() -> None:
    index = ExactVectorIndex.from_items({1: np.array([1.0, 0.0], dtype=np.float32)})

    assert index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=0) == []
    assert index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=-1) == []


def test_search_with_large_top_k_returns_all_vectors_sorted() -> None:
    index = ExactVectorIndex.from_items(
        {
            1: np.array([0.0, 1.0], dtype=np.float32),
            2: np.array([1.0, 0.0], dtype=np.float32),
            3: np.array([0.8, 0.2], dtype=np.float32),
        }
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=10)

    assert [item.vector_id for item in results] == [2, 3, 1]
    assert results[0].score > results[1].score > results[2].score


def test_search_subset_scores_only_selected_vector_ids() -> None:
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
            3: np.array([0.8, 0.2], dtype=np.float32),
        }
    )

    results = index.search_subset(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=5,
        candidate_vector_ids=[3],
    )

    assert [item.vector_id for item in results] == [3]


def test_search_uses_partial_top_k_selection(monkeypatch) -> None:
    index = ExactVectorIndex.from_items(
        {
            value: np.array([float(value), 1.0], dtype=np.float32)
            for value in range(1, 20)
        }
    )

    def fail_argsort(*_args, **_kwargs):
        raise AssertionError("search should not fully sort all scores for small top-k")

    monkeypatch.setattr(np, "argsort", fail_argsort)

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)

    assert len(results) == 3
    assert [item.score for item in results] == sorted(
        [item.score for item in results],
        reverse=True,
    )


def test_top_k_indices_rejects_multidimensional_scores() -> None:
    with pytest.raises(ValueError, match="scores must be one-dimensional"):
        top_k_indices(np.array([[1.0, 2.0]], dtype=np.float32), 1)


def test_zero_vectors_and_zero_query_return_finite_scores() -> None:
    index = ExactVectorIndex.from_items(
        {
            1: np.array([0.0, 0.0], dtype=np.float32),
            2: np.array([1.0, 0.0], dtype=np.float32),
        }
    )

    results = index.search(np.array([0.0, 0.0], dtype=np.float32), top_k=2)

    assert len(results) == 2
    assert all(np.isfinite(result.score) for result in results)
    assert all(result.score == 0.0 for result in results)

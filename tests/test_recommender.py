import numpy as np
import pytest

from paper_recommender.models import (
    DELETED_RECORD_MESSAGE,
    Paper,
    UNKNOWN_ID_MESSAGE,
    VECTOR_MISSING_MESSAGE,
)
from paper_recommender.recommender import RecommendationError, recommend
from paper_recommender.storage import connect_db, init_db, mark_deleted, upsert_paper
from paper_recommender.vector_store import ExactVectorIndex


def _paper(
    arxiv_id: str,
    vector_id: int | None,
    category: str,
    date: str | None = "2024-01-01",
    categories: tuple[str, ...] | None = None,
) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        vector_id=vector_id,
        active=True,
        oai_datestamp=date or "2024-01-01",
        published_date=date,
        updated_date=date,
        primary_category=category,
        categories=categories or (category,),
        content_hash=f"hash-{arxiv_id}",
    )


def _init_conn_with_papers(papers: list[Paper]):
    conn = connect_db(":memory:")
    init_db(conn)
    for paper in papers:
        upsert_paper(conn, paper)
    return conn


def test_recommend_excludes_query_paper_and_applies_top_k() -> None:
    conn = _init_conn_with_papers(
        [
            _paper("1706.03762", 1, "cs.CL"),
            _paper("1111.11111", 2, "cs.CL"),
            _paper("2222.22222", 3, "cs.LG"),
        ]
    )
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
            3: np.array([0.0, 1.0], dtype=np.float32),
        }
    )

    results = recommend(conn, index, "1706.03762", top_k=1)

    assert [result.arxiv_id for result in results] == ["1111.11111"]
    assert results[0].url == "https://arxiv.org/abs/1111.11111"
    assert results[0].similarity_score == pytest.approx(0.9938837)


def test_recommend_applies_category_filter() -> None:
    conn = _init_conn_with_papers(
        [
            _paper("1706.03762", 1, "cs.CL"),
            _paper("1111.11111", 2, "cs.CL"),
            _paper("2222.22222", 3, "cs.LG"),
        ]
    )
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
            3: np.array([0.8, 0.2], dtype=np.float32),
        }
    )

    results = recommend(conn, index, "1706.03762", top_k=5, category="cs.LG")

    assert [result.arxiv_id for result in results] == ["2222.22222"]


def test_recommend_category_filter_matches_secondary_category() -> None:
    conn = _init_conn_with_papers(
        [
            _paper("1706.03762", 1, "cs.CL"),
            _paper("1111.11111", 2, "cs.AI", categories=("cs.AI", "cs.LG")),
        ]
    )
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
        }
    )

    results = recommend(conn, index, "1706.03762", top_k=5, category="cs.LG")

    assert [result.arxiv_id for result in results] == ["1111.11111"]


def test_recommend_applies_multiple_category_filters_with_or_semantics() -> None:
    conn = _init_conn_with_papers(
        [
            _paper("1706.03762", 1, "cs.CL"),
            _paper("1111.11111", 2, "cs.AI"),
            _paper("2222.22222", 3, "cs.LG"),
            _paper("3333.33333", 4, "math.OC"),
        ]
    )
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.99, 0.01], dtype=np.float32),
            3: np.array([0.98, 0.02], dtype=np.float32),
            4: np.array([0.97, 0.03], dtype=np.float32),
        }
    )

    results = recommend(conn, index, "1706.03762", top_k=5, categories=["cs.AI", "math.OC"])

    assert [result.arxiv_id for result in results] == ["1111.11111", "3333.33333"]


def test_recommend_rejects_missing_paper() -> None:
    conn = _init_conn_with_papers([])
    index = ExactVectorIndex.from_items({1: np.array([1.0, 0.0], dtype=np.float32)})

    with pytest.raises(RecommendationError) as exc_info:
        recommend(conn, index, "missing", top_k=10)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == UNKNOWN_ID_MESSAGE


def test_recommend_rejects_vectorless_paper() -> None:
    conn = _init_conn_with_papers([_paper("1706.03762", None, "cs.CL")])
    index = ExactVectorIndex.from_items({1: np.array([1.0, 0.0], dtype=np.float32)})

    with pytest.raises(RecommendationError) as exc_info:
        recommend(conn, index, "1706.03762", top_k=10)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == VECTOR_MISSING_MESSAGE


def test_recommend_rejects_inactive_query_paper() -> None:
    conn = _init_conn_with_papers([_paper("1706.03762", 1, "cs.CL")])
    mark_deleted(conn, "1706.03762", "2024-01-02")
    index = ExactVectorIndex.from_items({1: np.array([1.0, 0.0], dtype=np.float32)})

    with pytest.raises(RecommendationError) as exc_info:
        recommend(conn, index, "1706.03762", top_k=10)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == DELETED_RECORD_MESSAGE


def test_recommend_excludes_inactive_candidates() -> None:
    conn = _init_conn_with_papers(
        [
            _paper("1706.03762", 1, "cs.CL"),
            _paper("1111.11111", 2, "cs.CL"),
            _paper("2222.22222", 3, "cs.CL"),
        ]
    )
    mark_deleted(conn, "1111.11111", "2024-01-02")
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.99, 0.01], dtype=np.float32),
            3: np.array([0.9, 0.1], dtype=np.float32),
        }
    )

    results = recommend(conn, index, "1706.03762", top_k=5)

    assert [result.arxiv_id for result in results] == ["2222.22222"]


def test_recommend_applies_published_date_filters() -> None:
    conn = _init_conn_with_papers(
        [
            _paper("1706.03762", 1, "cs.CL", "2024-01-15"),
            _paper("1111.11111", 2, "cs.CL", "2024-01-09"),
            _paper("2222.22222", 3, "cs.CL", "2024-01-10"),
            _paper("3333.33333", 4, "cs.CL", "2024-01-20"),
            _paper("4444.44444", 5, "cs.CL", "2024-01-21"),
        ]
    )
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.99, 0.01], dtype=np.float32),
            3: np.array([0.98, 0.02], dtype=np.float32),
            4: np.array([0.97, 0.03], dtype=np.float32),
            5: np.array([0.96, 0.04], dtype=np.float32),
        }
    )

    results = recommend(
        conn,
        index,
        "1706.03762",
        top_k=5,
        date_from="2024-01-10",
        date_to="2024-01-20",
    )

    assert [result.arxiv_id for result in results] == ["2222.22222", "3333.33333"]


def test_recommend_excludes_undated_candidate_when_date_filter_is_present() -> None:
    conn = _init_conn_with_papers(
        [
            _paper("1706.03762", 1, "cs.CL", "2024-01-15"),
            _paper("1111.11111", 2, "cs.CL", None),
            _paper("2222.22222", 3, "cs.CL", "2024-01-16"),
        ]
    )
    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.99, 0.01], dtype=np.float32),
            3: np.array([0.9, 0.1], dtype=np.float32),
        }
    )

    results_from = recommend(conn, index, "1706.03762", top_k=5, date_from="2024-01-01")
    results_to = recommend(conn, index, "1706.03762", top_k=5, date_to="2024-01-31")

    assert [result.arxiv_id for result in results_from] == ["2222.22222"]
    assert [result.arxiv_id for result in results_to] == ["2222.22222"]


def test_recommend_returns_empty_for_non_positive_top_k() -> None:
    conn = _init_conn_with_papers([_paper("1706.03762", 1, "cs.CL")])
    index = ExactVectorIndex.from_items({1: np.array([1.0, 0.0], dtype=np.float32)})

    assert recommend(conn, index, "1706.03762", top_k=0) == []
    assert recommend(conn, index, "1706.03762", top_k=-1) == []


def test_recommend_rejects_query_vector_missing_from_index() -> None:
    conn = _init_conn_with_papers([_paper("1706.03762", 1, "cs.CL")])
    index = ExactVectorIndex.from_items({2: np.array([1.0, 0.0], dtype=np.float32)})

    with pytest.raises(RecommendationError) as exc_info:
        recommend(conn, index, "1706.03762", top_k=10)

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == VECTOR_MISSING_MESSAGE


def test_recommendation_error_string_is_message() -> None:
    error = RecommendationError(404, "x")

    assert str(error) == "x"
    assert error.status_code == 404
    assert error.message == "x"

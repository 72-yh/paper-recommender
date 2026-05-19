import numpy as np
import pytest
from fastapi.testclient import TestClient

from paper_recommender.app import create_app
from paper_recommender.models import Paper, VECTOR_MISSING_MESSAGE
from paper_recommender.storage import connect_db, init_db, upsert_paper
from paper_recommender.vector_store import ExactVectorIndex


def _paper(
    arxiv_id: str,
    vector_id: int | None,
    category: str = "cs.CL",
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


def _build_client(tmp_path, papers: list[Paper] | None = None) -> TestClient:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    conn = connect_db(db_path)
    init_db(conn)
    for paper in papers or [
        _paper("1706.03762", 1, date="2017-06-12"),
        _paper("1111.11111", 2, date="2020-01-01"),
    ]:
        upsert_paper(conn, paper)
    conn.close()
    ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
        }
    ).save(index_path)
    return TestClient(create_app(db_path=db_path, index_path=index_path))


def test_health_endpoint(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_returns_results(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        "/api/recommend",
        json={"url": "https://arxiv.org/abs/1706.03762", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query_arxiv_id": "1706.03762",
        "results": [
            {
                "arxiv_id": "1111.11111",
                "url": "https://arxiv.org/abs/1111.11111",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "published_date": "2020-01-01",
                "updated_date": "2020-01-01",
                "similarity_score": pytest.approx(0.9938837),
            }
        ],
    }


def test_recommend_endpoint_rejects_invalid_url(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post("/api/recommend", json={"url": "https://example.com/nope"})

    assert response.status_code == 400
    assert "Please enter a valid arXiv URL" in response.json()["detail"]


def test_recommend_endpoint_maps_recommendation_error(tmp_path) -> None:
    client = _build_client(tmp_path, papers=[_paper("1706.03762", None)])

    response = client.post(
        "/api/recommend",
        json={"url": "https://arxiv.org/abs/1706.03762", "top_k": 1},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == VECTOR_MISSING_MESSAGE


def test_recommend_endpoint_maps_missing_index_vector(tmp_path) -> None:
    client = _build_client(tmp_path, papers=[_paper("3333.33333", 3)])

    response = client.post(
        "/api/recommend",
        json={"url": "https://arxiv.org/abs/3333.33333", "top_k": 1},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == VECTOR_MISSING_MESSAGE

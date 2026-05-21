import numpy as np
import pytest
from fastapi.testclient import TestClient

import paper_recommender.app as app_module
from paper_recommender.app import create_app
from paper_recommender.compressed_vector_store import Int8VectorIndex
from paper_recommender.models import Paper, VECTOR_MISSING_MESSAGE
from paper_recommender.storage import connect_db, init_db, set_pipeline_state, upsert_paper
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


def _build_int8_client(tmp_path) -> TestClient:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors_int8.npz"
    conn = connect_db(db_path)
    init_db(conn)
    for paper in [
        _paper("1706.03762", 1, date="2017-06-12"),
        _paper("1111.11111", 2, date="2020-01-01"),
    ]:
        upsert_paper(conn, paper)
    conn.close()
    exact = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
        }
    )
    Int8VectorIndex.from_exact_index(exact).save(index_path)
    return TestClient(create_app(db_path=db_path, index_path=index_path, index_kind="int8"))


def _assert_top_k_validation_error(
    response,
    expected_words: tuple[str, ...],
    expected_ctx: dict[str, int],
) -> None:
    matching_errors = [
        error
        for error in response.json()["detail"]
        if "top_k" in error.get("loc", ()) and all(
            word in error.get("msg", "").lower() for word in expected_words
        ) and all(error.get("ctx", {}).get(key) == value for key, value in expected_ctx.items())
    ]
    assert matching_errors


def test_health_endpoint(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint_returns_index_coverage(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors_int8.npz"
    conn = connect_db(db_path)
    init_db(conn)
    upsert_paper(conn, _paper("1706.03762", 1, date="2017-06-12"))
    upsert_paper(conn, _paper("1111.11111", 2, date="2020-01-01"))
    set_pipeline_state(conn, "last_successful_oai_datestamp", "2024-01-02")
    conn.close()
    Int8VectorIndex.from_exact_index(
        ExactVectorIndex.from_items(
            {
                1: np.array([1.0, 0.0], dtype=np.float32),
                2: np.array([0.9, 0.1], dtype=np.float32),
            }
        )
    ).save(index_path)
    client = TestClient(create_app(db_path=db_path, index_path=index_path, index_kind="int8"))

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json() == {
        "active_papers": 2,
        "indexed_papers": 2,
        "last_oai_datestamp": "2024-01-02",
        "index_kind": "int8",
        "index_bytes": index_path.stat().st_size,
    }


def test_module_exports_app() -> None:
    assert app_module.app is not None


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


def test_recommend_endpoint_can_use_int8_index(tmp_path) -> None:
    client = _build_int8_client(tmp_path)

    response = client.post(
        "/api/recommend",
        json={"url": "https://arxiv.org/abs/1706.03762", "top_k": 1},
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["arxiv_id"] == "1111.11111"
    assert result["similarity_score"] == pytest.approx(0.9938837, abs=0.01)


def test_create_app_reads_index_environment(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors_int8.npz"
    conn = connect_db(db_path)
    init_db(conn)
    upsert_paper(conn, _paper("1706.03762", 1, date="2017-06-12"))
    upsert_paper(conn, _paper("1111.11111", 2, date="2020-01-01"))
    conn.close()
    exact = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
        }
    )
    Int8VectorIndex.from_exact_index(exact).save(index_path)
    monkeypatch.setenv("PAPER_RECOMMENDER_DB_PATH", str(db_path))
    monkeypatch.setenv("PAPER_RECOMMENDER_INDEX_PATH", str(index_path))
    monkeypatch.setenv("PAPER_RECOMMENDER_INDEX_KIND", "int8")

    client = TestClient(create_app())
    response = client.post(
        "/api/recommend",
        json={"url": "https://arxiv.org/abs/1706.03762", "top_k": 1},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["arxiv_id"] == "1111.11111"


def test_recommend_endpoint_loads_index_once(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    conn = connect_db(db_path)
    init_db(conn)
    upsert_paper(conn, _paper("1706.03762", 1, date="2017-06-12"))
    upsert_paper(conn, _paper("1111.11111", 2, date="2020-01-01"))
    conn.close()
    ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1], dtype=np.float32),
        }
    ).save(index_path)
    load_count = 0
    original_load_index = app_module._load_index

    def counting_load_index(index_path, index_kind):
        nonlocal load_count
        load_count += 1
        return original_load_index(index_path, index_kind)

    monkeypatch.setattr(app_module, "_load_index", counting_load_index)
    client = TestClient(create_app(db_path=db_path, index_path=index_path))

    for _ in range(2):
        response = client.post(
            "/api/recommend",
            json={"url": "https://arxiv.org/abs/1706.03762", "top_k": 1},
        )
        assert response.status_code == 200

    assert load_count == 1


def test_recommend_endpoint_rejects_invalid_url(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post("/api/recommend", json={"url": "https://example.com/nope"})

    assert response.status_code == 400
    assert "Please enter a valid arXiv URL" in response.json()["detail"]


def test_recommend_endpoint_rejects_zero_top_k(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        "/api/recommend",
        json={"url": "https://arxiv.org/abs/1706.03762", "top_k": 0},
    )

    assert response.status_code == 422
    _assert_top_k_validation_error(response, ("greater", "equal"), {"ge": 1})


def test_recommend_endpoint_rejects_top_k_above_100(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        "/api/recommend",
        json={"url": "https://arxiv.org/abs/1706.03762", "top_k": 101},
    )

    assert response.status_code == 422
    _assert_top_k_validation_error(response, ("less", "equal"), {"le": 100})


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

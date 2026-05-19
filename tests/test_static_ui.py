from pathlib import Path

from fastapi.testclient import TestClient

from paper_recommender.app import create_app


STATIC_DIR = Path("src/paper_recommender/static")


def test_static_ui_uses_required_english_labels() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    for label in [
        "arXiv URL",
        "Category",
        "Date range",
        "Top K",
        "Find similar papers",
        "Similar papers",
        "Open on arXiv",
        "No results",
    ]:
        assert label in html


def test_static_ui_posts_to_recommend_endpoint() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/api/recommend"' in javascript


def test_static_ui_normalizes_non_success_error_details() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "async function parseJsonOrNull" in javascript
    assert "function normalizeErrorDetail" in javascript
    assert "Request failed" in javascript
    assert "Invalid request" in javascript
    assert ".json()" in javascript


def test_root_serves_static_ui_without_shadowing_api_routes(tmp_path) -> None:
    client = TestClient(
        create_app(
            db_path=tmp_path / "papers.db",
            index_path=tmp_path / "vectors.npz",
        )
    )

    root_response = client.get("/")
    health_response = client.get("/health")

    assert root_response.status_code == 200
    assert "text/html" in root_response.headers["content-type"]
    assert "Paper Recommender" in root_response.text
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

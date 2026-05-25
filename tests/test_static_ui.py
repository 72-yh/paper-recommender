from pathlib import Path
import re

from fastapi.testclient import TestClient

from paper_recommender.app import create_app


STATIC_DIR = Path("src/paper_recommender/static")


def test_static_ui_uses_required_english_labels() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    for label in [
        "arXiv URL",
        "Category",
        "Date range",
        "Find similar papers",
        "Similar papers",
        "Open on arXiv",
        "Index status unavailable",
        "No results",
    ]:
        assert label in html

    assert "Top K" not in html
    assert 'name="top_k"' not in html


def test_static_ui_posts_to_recommend_endpoint() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/api/recommend"' in javascript
    assert "top_k: 10" in javascript
    assert 'formData.get("top_k")' not in javascript


def test_static_ui_prevents_duplicate_recommendation_submits() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "const submitButton = form.querySelector" in javascript
    assert "submitButton.disabled = true" in javascript
    assert "submitButton.disabled = false" in javascript


def test_static_ui_fetches_index_status() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'fetch("/api/status")' in javascript
    assert "formatIndexStatus" in javascript
    assert "OAI through" in javascript


def test_static_ui_normalizes_non_success_error_details() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert re.search(
        r"async function parseJsonOrNull\(response\)\s*{.*?catch\s*{\s*return null;\s*}",
        javascript,
        re.DOTALL,
    )
    assert re.search(
        r"function normalizeErrorDetail\(detail\)\s*{"
        r".*?typeof detail === \"string\""
        r".*?return detail;"
        r".*?Array\.isArray\(detail\)"
        r".*?const itemWithMessage = detail\.find"
        r".*?typeof item\.msg === \"string\""
        r".*?return itemWithMessage\.msg;"
        r".*?return \"Invalid request\";",
        javascript,
        re.DOTALL,
    )


def test_static_ui_uses_request_failure_status_for_non_success_responses() -> None:
    javascript = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    non_ok_branch = re.search(
        r"if \(!response\.ok\)\s*{(?P<body>.*?)\n\s*}",
        javascript,
        re.DOTALL,
    )

    assert non_ok_branch is not None
    assert "errorStatusText(body)" in non_ok_branch.group("body")
    assert "No results" not in non_ok_branch.group("body")
    assert 'setStatus("Request failed")' in javascript


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

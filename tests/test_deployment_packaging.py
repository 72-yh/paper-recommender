from pathlib import Path


def test_dockerfile_packages_api_without_local_data() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "pip install --no-cache-dir ." in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "paper_recommender.app:app" in dockerfile
    assert "data/" not in dockerfile


def test_dockerignore_excludes_large_local_artifacts() -> None:
    ignored = Path(".dockerignore").read_text(encoding="utf-8").splitlines()

    for entry in [
        ".venv/",
        ".git/",
        ".pytest_cache/",
        ".ruff_cache/",
        "data/",
        "docs/evaluations/*.jsonl",
    ]:
        assert entry in ignored


def test_compose_mounts_data_and_uses_int8_index() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "paper-recommender:" in compose
    assert "PAPER_RECOMMENDER_DB_PATH: /app/data/paper_recommender_1m.db" in compose
    assert "PAPER_RECOMMENDER_INDEX_PATH: /app/data/vectors_1m_int8.npz" in compose
    assert "PAPER_RECOMMENDER_INDEX_KIND: int8" in compose
    assert "./data:/app/data:ro" in compose
    assert '"8000:8000"' in compose


def test_readme_documents_container_deployment() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Container Deployment" in readme
    assert "docker compose up --build" in readme
    assert "scripts\\preflight_artifacts.py" in readme
    assert "scripts\\smoke_deployment.py" in readme
    assert "data/paper_recommender_1m.db" in readme
    assert "data/vectors_1m_int8.npz" in readme

from pathlib import Path
import tomllib


def test_dockerfile_packages_api_without_local_data() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "pip install --no-cache-dir ." in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert "paper_recommender.app:app" in dockerfile
    assert "data/" not in dockerfile


def test_dockerfile_defines_python_healthcheck() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "HEALTHCHECK" in dockerfile
    assert "urllib.request" in dockerfile
    assert "http://127.0.0.1:8000/health" in dockerfile
    assert "curl" not in dockerfile.lower()


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


def test_compose_uses_product_project_name() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "name: paper_recommender" in compose
    assert "name: arxiv" not in compose.lower()


def test_compose_defines_runtime_healthcheck() -> None:
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")

    assert "healthcheck:" in compose
    assert "http://127.0.0.1:8000/health" in compose
    assert "interval: 30s" in compose
    assert "start_period: 20s" in compose


def test_readme_documents_container_deployment() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Container Deployment" in readme
    assert "docker compose up --build" in readme
    assert "scripts\\verify_container_deployment.py" in readme
    assert "Docker healthcheck" in readme
    assert "scripts\\preflight_artifacts.py" in readme
    assert "scripts\\smoke_deployment.py" in readme
    assert "data/paper_recommender_1m.db" in readme
    assert "data/vectors_1m_int8.npz" in readme
    assert "Compose project name is fixed to `paper_recommender`" in readme


def test_fly_config_uses_low_cost_single_machine() -> None:
    config = tomllib.loads(Path("fly.toml").read_text(encoding="utf-8"))

    assert config["app"].startswith("paper-recommender")
    assert "arxiv" not in config["app"]

    service = config["http_service"]
    assert service["internal_port"] == 8000
    assert service["auto_stop_machines"] == "stop"
    assert service["auto_start_machines"] is True
    assert service["min_machines_running"] == 0

    vm = config["vm"][0]
    assert vm["cpu_kind"] == "shared"
    assert vm["cpus"] == 1
    assert vm["memory_mb"] == 1024

    mount = config["mounts"][0]
    assert mount["source"] == "paper_recommender_data"
    assert mount["destination"] == "/app/data"
    assert mount["initial_size"] == "2GB"
    assert mount["scheduled_snapshots"] is False


def test_fly_config_avoids_known_cost_multipliers() -> None:
    fly_config = Path("fly.toml").read_text(encoding="utf-8").lower()

    assert "autoscaler" not in fly_config
    assert "dedicated" not in fly_config
    assert "allocate-v4" not in fly_config


def test_fly_runbook_documents_cost_guardrails() -> None:
    runbook = Path("docs/deployment/fly-low-cost.md").read_text(encoding="utf-8")

    assert "Cost Guardrails" in runbook
    assert "shared-cpu-1x" in runbook
    assert "1GB RAM" in runbook
    assert "Do not run `fly ips allocate-v4`" in runbook
    assert "Do not enable metrics-based autoscaling" in runbook
    assert "Do not create Managed Postgres, Redis, Tigris, or other managed services" in runbook
    assert "Check Fly Dashboard > Billing" in runbook

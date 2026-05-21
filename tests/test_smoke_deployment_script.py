import subprocess
import sys

import pytest

from scripts.smoke_deployment import DeploymentSmokeError, smoke_deployment


def test_smoke_deployment_checks_health_status_and_recommendation() -> None:
    calls: list[tuple[str, str]] = []

    def http_get(url: str):
        calls.append(("GET", url))
        if url.endswith("/health"):
            return {"status": "ok"}
        if url.endswith("/api/status"):
            return {
                "active_papers": 1_000_000,
                "indexed_papers": 1_000_000,
                "last_oai_datestamp": "2016-01-27",
                "index_kind": "int8",
                "index_bytes": 340_151_071,
            }
        raise AssertionError(f"unexpected GET {url}")

    def http_post(url: str, payload: dict[str, object]):
        calls.append(("POST", url))
        assert payload == {
            "url": "https://arxiv.org/abs/0704.0004",
            "top_k": 3,
        }
        return {
            "query_arxiv_id": "0704.0004",
            "results": [{"arxiv_id": "1001.2508"}],
        }

    summary = smoke_deployment(
        base_url="http://example.test/",
        query_url="https://arxiv.org/abs/0704.0004",
        top_k=3,
        min_indexed_papers=1_000,
        expected_index_kind="int8",
        http_get=http_get,
        http_post=http_post,
    )

    assert calls == [
        ("GET", "http://example.test/health"),
        ("GET", "http://example.test/api/status"),
        ("POST", "http://example.test/api/recommend"),
    ]
    assert summary.indexed_papers == 1_000_000
    assert summary.result_count == 1
    assert summary.last_oai_datestamp == "2016-01-27"


def test_smoke_deployment_rejects_unhealthy_service() -> None:
    def http_get(_url: str):
        return {"status": "starting"}

    with pytest.raises(DeploymentSmokeError, match="Health check failed"):
        smoke_deployment(
            base_url="http://example.test",
            query_url="https://arxiv.org/abs/0704.0004",
            http_get=http_get,
            http_post=lambda _url, _payload: {},
        )


def test_smoke_deployment_rejects_small_or_wrong_index() -> None:
    responses = {
        "/health": {"status": "ok"},
        "/api/status": {
            "active_papers": 10,
            "indexed_papers": 10,
            "last_oai_datestamp": "2016-01-27",
            "index_kind": "exact",
            "index_bytes": 100,
        },
    }

    def http_get(url: str):
        for suffix, response in responses.items():
            if url.endswith(suffix):
                return response
        raise AssertionError(f"unexpected GET {url}")

    with pytest.raises(DeploymentSmokeError, match="Expected index kind int8"):
        smoke_deployment(
            base_url="http://example.test",
            query_url="https://arxiv.org/abs/0704.0004",
            expected_index_kind="int8",
            http_get=http_get,
            http_post=lambda _url, _payload: {"results": [{"arxiv_id": "x"}]},
        )

    with pytest.raises(DeploymentSmokeError, match="Indexed paper count"):
        smoke_deployment(
            base_url="http://example.test",
            query_url="https://arxiv.org/abs/0704.0004",
            min_indexed_papers=1_000,
            http_get=http_get,
            http_post=lambda _url, _payload: {"results": [{"arxiv_id": "x"}]},
        )


def test_smoke_deployment_rejects_empty_recommendations() -> None:
    def http_get(url: str):
        if url.endswith("/health"):
            return {"status": "ok"}
        return {"indexed_papers": 1_000, "index_kind": "int8"}

    with pytest.raises(DeploymentSmokeError, match="Recommendation check returned no results"):
        smoke_deployment(
            base_url="http://example.test",
            query_url="https://arxiv.org/abs/0704.0004",
            http_get=http_get,
            http_post=lambda _url, _payload: {"results": []},
        )


def test_smoke_deployment_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/smoke_deployment.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Smoke test a running Paper Recommender deployment" in result.stdout

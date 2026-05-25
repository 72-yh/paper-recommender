import subprocess
import sys
from pathlib import Path

import pytest

from scripts.preflight_artifacts import ArtifactPreflightSummary
from scripts.smoke_deployment import DeploymentSmokeSummary
from scripts.verify_container_deployment import (
    CommandResult,
    ContainerVerificationError,
    run_subprocess,
    verify_container_deployment,
)


def test_verify_container_deployment_runs_expected_sequence(tmp_path) -> None:
    calls: list[tuple[str, object]] = []

    def preflight(**kwargs):
        calls.append(("preflight", kwargs))
        return _preflight_summary(kwargs["db_path"], kwargs["index_path"])

    def run_command(command):
        calls.append(("command", tuple(command)))
        return CommandResult(tuple(command), 0, "", "")

    def health_checker(**kwargs):
        calls.append(("health", kwargs["base_url"]))

    def smoke(**kwargs):
        calls.append(("smoke", kwargs))
        return _smoke_summary(kwargs["base_url"])

    summary = verify_container_deployment(
        db_path=tmp_path / "papers.db",
        index_path=tmp_path / "vectors.npz",
        index_kind="int8",
        min_indexed_papers=1_000,
        base_url="http://example.test",
        query_url="https://arxiv.org/abs/0704.0004",
        top_k=3,
        docker_command="docker",
        run_command=run_command,
        preflight=preflight,
        health_checker=health_checker,
        smoke=smoke,
    )

    assert [call[0] for call in calls] == [
        "preflight",
        "command",
        "command",
        "command",
        "health",
        "smoke",
    ]
    assert calls[1] == ("command", ("docker", "compose", "config"))
    assert calls[2] == ("command", ("docker", "compose", "build"))
    assert calls[3] == ("command", ("docker", "compose", "up", "-d"))
    assert summary.preflight.indexed_papers == 1_000_000
    assert summary.smoke.result_count == 3
    assert summary.commands_run == [
        ("docker", "compose", "config"),
        ("docker", "compose", "build"),
        ("docker", "compose", "up", "-d"),
    ]


def test_verify_container_deployment_supports_compose_file_and_skip_flags(tmp_path) -> None:
    commands: list[tuple[str, ...]] = []

    def run_command(command):
        commands.append(tuple(command))
        return CommandResult(tuple(command), 0, "", "")

    verify_container_deployment(
        db_path=tmp_path / "papers.db",
        index_path=tmp_path / "vectors.npz",
        compose_file=Path("docker-compose.prod.yml"),
        build=False,
        up=False,
        run_command=run_command,
        preflight=lambda **kwargs: _preflight_summary(kwargs["db_path"], kwargs["index_path"]),
        health_checker=lambda **_kwargs: None,
        smoke=lambda **kwargs: _smoke_summary(kwargs["base_url"]),
    )

    assert commands == [("docker", "compose", "-f", "docker-compose.prod.yml", "config")]


def test_verify_container_deployment_rejects_failed_compose_command(tmp_path) -> None:
    def run_command(command):
        return CommandResult(tuple(command), 17, "", "compose failed")

    with pytest.raises(ContainerVerificationError, match="docker compose config"):
        verify_container_deployment(
            db_path=tmp_path / "papers.db",
            index_path=tmp_path / "vectors.npz",
            run_command=run_command,
            preflight=lambda **kwargs: _preflight_summary(kwargs["db_path"], kwargs["index_path"]),
            health_checker=lambda **_kwargs: None,
            smoke=lambda **kwargs: _smoke_summary(kwargs["base_url"]),
        )


def test_run_subprocess_decodes_docker_output_as_utf8_with_replacement(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("scripts.verify_container_deployment.subprocess.run", fake_run)

    result = run_subprocess(("docker", "compose", "build"))

    assert result.stdout == "ok"
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_verify_container_deployment_cli_help_loads() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_container_deployment.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Verify a local Paper Recommender container deployment" in result.stdout


def _preflight_summary(db_path, index_path) -> ArtifactPreflightSummary:
    return ArtifactPreflightSummary(
        db_path=Path(db_path),
        index_path=Path(index_path),
        index_kind="int8",
        active_papers=1_000_000,
        indexed_papers=1_000_000,
        index_vectors=1_000_000,
        dimensions=384,
        db_bytes=411_156_480,
        index_bytes=340_151_071,
        total_artifact_bytes=751_307_551,
        last_oai_datestamp="2016-01-27",
        vector_ids_checked=True,
        category_lookup_checked=True,
        category_lookup_rows=1_558_846,
        target_indexed_papers=None,
        projected_total_artifact_bytes=None,
        max_volume_gb=None,
    )


def _smoke_summary(base_url: str) -> DeploymentSmokeSummary:
    return DeploymentSmokeSummary(
        base_url=base_url,
        indexed_papers=1_000_000,
        last_oai_datestamp="2016-01-27",
        index_kind="int8",
        result_count=3,
        query_arxiv_id="0704.0004",
    )

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

try:
    from scripts.preflight_artifacts import (
        ArtifactPreflightError,
        ArtifactPreflightSummary,
        preflight_artifacts,
    )
    from scripts.smoke_deployment import (
        DeploymentSmokeError,
        DeploymentSmokeSummary,
        http_get_json,
        smoke_deployment,
    )
except ModuleNotFoundError:
    from preflight_artifacts import (
        ArtifactPreflightError,
        ArtifactPreflightSummary,
        preflight_artifacts,
    )
    from smoke_deployment import (
        DeploymentSmokeError,
        DeploymentSmokeSummary,
        http_get_json,
        smoke_deployment,
    )


Command = Sequence[str]
CommandRunner = Callable[[Command], "CommandResult"]
PreflightRunner = Callable[..., ArtifactPreflightSummary]
HealthChecker = Callable[..., None]
SmokeRunner = Callable[..., DeploymentSmokeSummary]


class ContainerVerificationError(Exception):
    """Raised when container deployment verification fails."""


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ContainerVerificationSummary:
    preflight: ArtifactPreflightSummary
    smoke: DeploymentSmokeSummary
    commands_run: list[tuple[str, ...]]


def verify_container_deployment(
    *,
    db_path: str | Path = Path("data/paper_recommender_1m.db"),
    index_path: str | Path = Path("data/vectors_1m_int8.npz"),
    index_kind: str = "int8",
    min_indexed_papers: int = 1_000_000,
    base_url: str = "http://127.0.0.1:8000",
    query_url: str = "https://arxiv.org/abs/0704.0004",
    top_k: int = 3,
    docker_command: str = "docker",
    compose_file: str | Path | None = None,
    build: bool = True,
    up: bool = True,
    wait_timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 2.0,
    run_command: CommandRunner = None,
    preflight: PreflightRunner = preflight_artifacts,
    health_checker: HealthChecker = None,
    smoke: SmokeRunner = smoke_deployment,
) -> ContainerVerificationSummary:
    run_command = run_command or run_subprocess
    health_checker = health_checker or wait_for_health

    try:
        preflight_summary = preflight(
            db_path=db_path,
            index_path=index_path,
            index_kind=index_kind,
            min_indexed_papers=min_indexed_papers,
        )
    except ArtifactPreflightError as exc:
        raise ContainerVerificationError(str(exc)) from exc

    commands_run: list[tuple[str, ...]] = []
    for command in _compose_commands(
        docker_command=docker_command,
        compose_file=compose_file,
        build=build,
        up=up,
    ):
        result = _run_required(command, run_command)
        commands_run.append(result.command)

    health_checker(
        base_url=base_url,
        timeout_seconds=wait_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    try:
        smoke_summary = smoke(
            base_url=base_url,
            query_url=query_url,
            top_k=top_k,
            min_indexed_papers=min_indexed_papers,
            expected_index_kind=index_kind,
        )
    except DeploymentSmokeError as exc:
        raise ContainerVerificationError(str(exc)) from exc

    return ContainerVerificationSummary(
        preflight=preflight_summary,
        smoke=smoke_summary,
        commands_run=commands_run,
    )


def run_subprocess(command: Command) -> CommandResult:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    return CommandResult(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def wait_for_health(
    *,
    base_url: str,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 2.0,
    http_get=http_get_json,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> None:
    base_url = base_url.rstrip("/")
    deadline = monotonic() + timeout_seconds
    last_error: Exception | None = None
    last_response: object = None

    while True:
        try:
            response = http_get(f"{base_url}/health")
            last_response = response
            if response.get("status") == "ok":
                return
        except Exception as exc:  # noqa: BLE001 - this is a readiness poll boundary.
            last_error = exc

        if monotonic() >= deadline:
            detail = f"last_error={last_error}" if last_error else f"last_response={last_response}"
            raise ContainerVerificationError(
                f"Health check did not become ready within {timeout_seconds:g}s ({detail})"
            )
        sleep(poll_interval_seconds)


def _compose_commands(
    *,
    docker_command: str,
    compose_file: str | Path | None,
    build: bool,
    up: bool,
) -> list[tuple[str, ...]]:
    base = [docker_command, "compose"]
    if compose_file is not None:
        base.extend(["-f", str(compose_file)])

    commands = [tuple([*base, "config"])]
    if build:
        commands.append(tuple([*base, "build"]))
    if up:
        commands.append(tuple([*base, "up", "-d"]))
    return commands


def _run_required(command: Command, run_command: CommandRunner) -> CommandResult:
    result = run_command(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ContainerVerificationError(
            f"Command failed ({' '.join(result.command)}): {detail}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a local Paper Recommender container deployment."
    )
    parser.add_argument("--db-path", type=Path, default=Path("data/paper_recommender_1m.db"))
    parser.add_argument("--index-path", type=Path, default=Path("data/vectors_1m_int8.npz"))
    parser.add_argument("--index-kind", choices=("exact", "int8"), default="int8")
    parser.add_argument("--min-indexed-papers", type=int, default=1_000_000)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query-url", default="https://arxiv.org/abs/0704.0004")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--docker-command", default="docker")
    parser.add_argument("--compose-file", type=Path)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-up", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    args = parser.parse_args()

    try:
        summary = verify_container_deployment(
            db_path=args.db_path,
            index_path=args.index_path,
            index_kind=args.index_kind,
            min_indexed_papers=args.min_indexed_papers,
            base_url=args.base_url,
            query_url=args.query_url,
            top_k=args.top_k,
            docker_command=args.docker_command,
            compose_file=args.compose_file,
            build=not args.skip_build,
            up=not args.skip_up,
            wait_timeout_seconds=args.wait_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except ContainerVerificationError as exc:
        print(f"Container verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        "Container verification ok: "
        f"commands={len(summary.commands_run)} "
        f"indexed_papers={summary.preflight.indexed_papers} "
        f"index_kind={summary.preflight.index_kind} "
        f"dimensions={summary.preflight.dimensions} "
        f"last_oai_datestamp={summary.preflight.last_oai_datestamp} "
        f"result_count={summary.smoke.result_count} "
        f"query_arxiv_id={summary.smoke.query_arxiv_id}"
    )


if __name__ == "__main__":
    main()

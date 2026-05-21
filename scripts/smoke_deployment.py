from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


JsonObject = dict[str, object]
HttpGet = Callable[[str], JsonObject]
HttpPost = Callable[[str, JsonObject], JsonObject]


class DeploymentSmokeError(Exception):
    """Raised when a deployment smoke check fails."""


@dataclass(frozen=True)
class DeploymentSmokeSummary:
    base_url: str
    indexed_papers: int
    last_oai_datestamp: str | None
    index_kind: str | None
    result_count: int
    query_arxiv_id: str | None


def smoke_deployment(
    *,
    base_url: str,
    query_url: str,
    top_k: int = 3,
    min_indexed_papers: int = 1,
    expected_index_kind: str | None = None,
    http_get: HttpGet | None = None,
    http_post: HttpPost | None = None,
) -> DeploymentSmokeSummary:
    http_get = http_get or http_get_json
    http_post = http_post or http_post_json
    base_url = base_url.rstrip("/")

    health = http_get(_join_url(base_url, "/health"))
    if health.get("status") != "ok":
        raise DeploymentSmokeError(f"Health check failed: {health}")

    status = http_get(_join_url(base_url, "/api/status"))
    index_kind = _optional_str(status.get("index_kind"))
    if expected_index_kind is not None and index_kind != expected_index_kind:
        raise DeploymentSmokeError(f"Expected index kind {expected_index_kind}, got {index_kind}")

    indexed_papers = _int_field(status, "indexed_papers")
    if indexed_papers < min_indexed_papers:
        raise DeploymentSmokeError(
            f"Indexed paper count {indexed_papers} is below minimum {min_indexed_papers}"
        )

    recommendation = http_post(
        _join_url(base_url, "/api/recommend"),
        {
            "url": query_url,
            "top_k": top_k,
        },
    )
    results = recommendation.get("results")
    if not isinstance(results, list) or not results:
        raise DeploymentSmokeError("Recommendation check returned no results")

    return DeploymentSmokeSummary(
        base_url=base_url,
        indexed_papers=indexed_papers,
        last_oai_datestamp=_optional_str(status.get("last_oai_datestamp")),
        index_kind=index_kind,
        result_count=len(results),
        query_arxiv_id=_optional_str(recommendation.get("query_arxiv_id")),
    )


def http_get_json(url: str) -> JsonObject:
    request = Request(url, headers={"Accept": "application/json"})
    return _request_json(request)


def http_post_json(url: str, payload: JsonObject) -> JsonObject:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    return _request_json(request)


def _request_json(request: Request) -> JsonObject:
    try:
        with urlopen(request, timeout=30) as response:
            data = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DeploymentSmokeError(f"HTTP {exc.code} for {request.full_url}: {detail}") from exc
    except URLError as exc:
        raise DeploymentSmokeError(f"Could not reach {request.full_url}: {exc.reason}") from exc

    try:
        decoded = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise DeploymentSmokeError(f"Invalid JSON from {request.full_url}") from exc
    if not isinstance(decoded, dict):
        raise DeploymentSmokeError(f"Expected JSON object from {request.full_url}")
    return decoded


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url}{path}"


def _int_field(data: JsonObject, name: str) -> int:
    value = data.get(name)
    if not isinstance(value, int):
        raise DeploymentSmokeError(f"Expected integer field {name}, got {value!r}")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test a running Paper Recommender deployment."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query-url", default="https://arxiv.org/abs/0704.0004")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-indexed-papers", type=int, default=1)
    parser.add_argument("--expected-index-kind")
    args = parser.parse_args()

    try:
        summary = smoke_deployment(
            base_url=args.base_url,
            query_url=args.query_url,
            top_k=args.top_k,
            min_indexed_papers=args.min_indexed_papers,
            expected_index_kind=args.expected_index_kind,
        )
    except DeploymentSmokeError as exc:
        print(f"Deployment smoke failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        "Deployment smoke ok: "
        f"base_url={summary.base_url} "
        f"indexed_papers={summary.indexed_papers} "
        f"index_kind={summary.index_kind} "
        f"last_oai_datestamp={summary.last_oai_datestamp} "
        f"result_count={summary.result_count} "
        f"query_arxiv_id={summary.query_arxiv_id}"
    )


if __name__ == "__main__":
    main()

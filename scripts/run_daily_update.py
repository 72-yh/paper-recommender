from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from paper_recommender.embedding import DEFAULT_MODEL_NAME
from paper_recommender.oai_client import (
    DEFAULT_FETCH_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
    OAI_ENDPOINT,
)

try:
    from scripts.build_ivf_int8_index import IvfBuildReport, build_ivf_int8_index
    from scripts.preflight_artifacts import ArtifactPreflightSummary, preflight_artifacts
    from scripts.sync_serving_index import ServingSyncSummary, sync_serving_index
except ModuleNotFoundError:
    from build_ivf_int8_index import IvfBuildReport, build_ivf_int8_index
    from preflight_artifacts import ArtifactPreflightSummary, preflight_artifacts
    from sync_serving_index import ServingSyncSummary, sync_serving_index


@dataclass(frozen=True)
class DailyUpdateSummary:
    sync: ServingSyncSummary
    rebuilt_ivf: bool
    ivf: IvfBuildReport | None
    preflight: ArtifactPreflightSummary | None


def run_daily_update(
    *,
    db_path: str | Path = Path("data/paper_recommender_1m.db"),
    exact_index_path: str | Path = Path("data/vectors_1m.npz"),
    serving_index_path: str | Path = Path("data/vectors_1m_int8_mmap"),
    endpoint: str = OAI_ENDPOINT,
    from_date: str | None = None,
    until_date: str | None = None,
    batch_limit: int | None = None,
    max_records: int | None = None,
    embedder_backend: str = "sentence-transformers",
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    dimensions: int = 384,
    request_delay_seconds: float = 3.0,
    fetch_retries: int = DEFAULT_FETCH_RETRIES,
    fetch_retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    checkpoint_every_records: int | None = 10_000,
    embedding_batch_size: int = 128,
    target_vector_count: int | None = None,
    sample_size: int = 1_000,
    label: str = "daily-ivf-int8-mmap",
    jsonl_report_path: str | Path = "docs/evaluations/compression-runs.jsonl",
    markdown_report_path: str | Path = "docs/evaluations/compression-runs.md",
    force_rebuild: bool = False,
    force_ivf_rebuild: bool = False,
    record_report: bool = True,
    n_clusters: int = 512,
    train_sample_size: int = 100_000,
    ivf_iterations: int = 6,
    assignment_batch_size: int = 65_536,
    seed: int = 13,
    run_preflight: bool = True,
    min_indexed_papers: int = 3_000_000,
    target_indexed_papers: int | None = 3_000_000,
    max_volume_gb: float | None = 4.0,
    sync: Callable[..., ServingSyncSummary] = sync_serving_index,
    build_ivf: Callable[..., IvfBuildReport] = build_ivf_int8_index,
    preflight: Callable[..., ArtifactPreflightSummary] = preflight_artifacts,
) -> DailyUpdateSummary:
    db_path = Path(db_path)
    exact_index_path = Path(exact_index_path)
    serving_index_path = Path(serving_index_path)

    sync_summary = sync(
        endpoint=endpoint,
        db_path=db_path,
        exact_index_path=exact_index_path,
        serving_index_path=serving_index_path,
        serving_index_kind="int8_mmap",
        from_date=from_date,
        until_date=until_date,
        batch_limit=batch_limit,
        max_records=max_records,
        embedder_backend=embedder_backend,
        model_name=model_name,
        device=device,
        dimensions=dimensions,
        request_delay_seconds=request_delay_seconds,
        fetch_retries=fetch_retries,
        fetch_retry_delay_seconds=fetch_retry_delay_seconds,
        checkpoint_every_records=checkpoint_every_records,
        embedding_batch_size=embedding_batch_size,
        target_vector_count=target_vector_count,
        compression_method="int8",
        pca_dimensions=None,
        top_k=10,
        sample_size=sample_size,
        label=label,
        jsonl_report_path=jsonl_report_path,
        markdown_report_path=markdown_report_path,
        force_rebuild=force_rebuild,
        record_report=record_report,
    )

    ivf_report = None
    if sync_summary.rebuilt_serving_index or force_ivf_rebuild:
        ivf_report = build_ivf(
            index_path=serving_index_path,
            n_clusters=n_clusters,
            train_sample_size=train_sample_size,
            iterations=ivf_iterations,
            assignment_batch_size=assignment_batch_size,
            seed=seed,
        )

    preflight_summary = None
    if run_preflight:
        preflight_summary = preflight(
            db_path=db_path,
            index_path=serving_index_path,
            index_kind="ivf_int8_mmap",
            min_indexed_papers=min_indexed_papers,
            check_vector_ids=True,
            check_category_lookup=True,
            target_indexed_papers=target_indexed_papers,
            max_volume_gb=max_volume_gb,
        )

    return DailyUpdateSummary(
        sync=sync_summary,
        rebuilt_ivf=ivf_report is not None,
        ivf=ivf_report,
        preflight=preflight_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run daily OAI sync, refresh local serving artifacts, rebuild IVF, and preflight."
    )
    parser.add_argument("--endpoint", default=OAI_ENDPOINT)
    parser.add_argument("--from-date", dest="from_date")
    parser.add_argument("--until-date", dest="until_date")
    parser.add_argument("--max-batches", dest="batch_limit", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--embedder", choices=("sentence-transformers", "hashing"), default="sentence-transformers")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--dimensions", type=int, default=384)
    parser.add_argument("--request-delay-seconds", type=float, default=3.0)
    parser.add_argument("--fetch-retries", type=int, default=DEFAULT_FETCH_RETRIES)
    parser.add_argument("--fetch-retry-delay-seconds", type=float, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--checkpoint-every-records", type=int, default=10_000)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--target-vector-count", type=int)
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--label", default="daily-ivf-int8-mmap")
    parser.add_argument("--db-path", type=Path, default=Path("data/paper_recommender_1m.db"))
    parser.add_argument("--exact-index-path", type=Path, default=Path("data/vectors_1m.npz"))
    parser.add_argument("--serving-index-path", type=Path, default=Path("data/vectors_1m_int8_mmap"))
    parser.add_argument("--jsonl-report", type=Path, default=Path("docs/evaluations/compression-runs.jsonl"))
    parser.add_argument("--markdown-report", type=Path, default=Path("docs/evaluations/compression-runs.md"))
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--force-ivf-rebuild", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=512)
    parser.add_argument("--train-sample-size", type=int, default=100_000)
    parser.add_argument("--ivf-iterations", type=int, default=6)
    parser.add_argument("--assignment-batch-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--min-indexed-papers", type=int, default=3_000_000)
    parser.add_argument("--target-indexed-papers", type=int, default=3_000_000)
    parser.add_argument("--max-volume-gb", type=float, default=4.0)
    args = parser.parse_args()

    summary = run_daily_update(
        endpoint=args.endpoint,
        db_path=args.db_path,
        exact_index_path=args.exact_index_path,
        serving_index_path=args.serving_index_path,
        from_date=args.from_date,
        until_date=args.until_date,
        batch_limit=args.batch_limit,
        max_records=args.max_records,
        embedder_backend=args.embedder,
        model_name=args.model_name,
        device=args.device,
        dimensions=args.dimensions,
        request_delay_seconds=args.request_delay_seconds,
        fetch_retries=args.fetch_retries,
        fetch_retry_delay_seconds=args.fetch_retry_delay_seconds,
        checkpoint_every_records=args.checkpoint_every_records,
        embedding_batch_size=args.embedding_batch_size,
        target_vector_count=args.target_vector_count,
        sample_size=args.sample_size,
        label=args.label,
        jsonl_report_path=args.jsonl_report,
        markdown_report_path=args.markdown_report,
        force_rebuild=args.force_rebuild,
        force_ivf_rebuild=args.force_ivf_rebuild,
        record_report=not args.no_record,
        n_clusters=args.n_clusters,
        train_sample_size=args.train_sample_size,
        ivf_iterations=args.ivf_iterations,
        assignment_batch_size=args.assignment_batch_size,
        seed=args.seed,
        run_preflight=not args.skip_preflight,
        min_indexed_papers=args.min_indexed_papers,
        target_indexed_papers=args.target_indexed_papers,
        max_volume_gb=args.max_volume_gb,
    )
    print(_format_summary(summary))


def _format_summary(summary: DailyUpdateSummary) -> str:
    sync = summary.sync
    update = sync.update
    parts = [
        "Daily update:",
        f"records={update.records_seen}",
        f"embedded={update.embedded}",
        f"deleted={update.deleted}",
        f"last_datestamp={update.last_datestamp}",
        f"rebuilt_serving_index={sync.rebuilt_serving_index}",
        f"rebuilt_ivf={summary.rebuilt_ivf}",
    ]
    if summary.ivf is not None:
        parts.extend(
            [
                f"ivf_clusters={summary.ivf.n_clusters}",
                f"ivf_output_bytes={summary.ivf.output_bytes}",
            ]
        )
    if summary.preflight is not None:
        parts.extend(
            [
                f"indexed_papers={summary.preflight.indexed_papers}",
                f"index_kind={summary.preflight.index_kind}",
                f"total_artifact_bytes={summary.preflight.total_artifact_bytes}",
                f"max_volume_gb={summary.preflight.max_volume_gb}",
            ]
        )
    return " ".join(parts)


if __name__ == "__main__":
    main()

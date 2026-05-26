from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from paper_recommender.embedding import DEFAULT_MODEL_NAME
from paper_recommender.index_builder import IndexBuildSummary, build_index_from_oai
from paper_recommender.oai_client import (
    DEFAULT_FETCH_RETRIES,
    DEFAULT_RETRY_DELAY_SECONDS,
    OAI_ENDPOINT,
)

try:
    from scripts.evaluate_compression import CompressionReport, append_report, evaluate_compression
except ModuleNotFoundError:
    from evaluate_compression import CompressionReport, append_report, evaluate_compression

try:
    from scripts.convert_int8_mmap import convert_int8_index_to_mmap
except ModuleNotFoundError:
    from convert_int8_mmap import convert_int8_index_to_mmap


@dataclass(frozen=True)
class ServingSyncSummary:
    update: IndexBuildSummary
    serving_index_kind: str
    serving_index_path: Path
    rebuilt_serving_index: bool
    compression: CompressionReport | None


def sync_serving_index(
    *,
    db_path: str | Path,
    exact_index_path: str | Path,
    serving_index_path: str | Path,
    serving_index_kind: str = "int8",
    endpoint: str = OAI_ENDPOINT,
    from_date: str | None = None,
    until_date: str | None = None,
    batch_limit: int | None = None,
    max_records: int | None = None,
    embedder_backend: str = "sentence-transformers",
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    dimensions: int = 256,
    request_delay_seconds: float = 3.0,
    fetch_retries: int = DEFAULT_FETCH_RETRIES,
    fetch_retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    checkpoint_every_records: int | None = 10_000,
    embedding_batch_size: int = 128,
    compression_method: str = "int8",
    pca_dimensions: int | None = None,
    top_k: int = 10,
    sample_size: int = 1_000,
    label: str = "daily-int8",
    jsonl_report_path: str | Path = "docs/evaluations/compression-runs.jsonl",
    markdown_report_path: str | Path = "docs/evaluations/compression-runs.md",
    force_rebuild: bool = False,
    record_report: bool = True,
    update_index: Callable[..., IndexBuildSummary] = build_index_from_oai,
    evaluate: Callable[..., CompressionReport] = evaluate_compression,
) -> ServingSyncSummary:
    serving_index_path = Path(serving_index_path)
    if serving_index_kind not in {"int8", "int8_mmap"}:
        raise ValueError(f"unsupported serving index kind: {serving_index_kind}")

    update = update_index(
        endpoint=endpoint,
        db_path=db_path,
        index_path=exact_index_path,
        from_date=from_date,
        until_date=until_date,
        batch_limit=batch_limit,
        max_records=max_records,
        embedder_backend=embedder_backend,
        model_name=model_name,
        device=device,
        dimensions=dimensions,
        resume=True,
        request_delay_seconds=request_delay_seconds,
        fetch_retries=fetch_retries,
        fetch_retry_delay_seconds=fetch_retry_delay_seconds,
        checkpoint_every_records=checkpoint_every_records,
        embedding_batch_size=embedding_batch_size,
    )

    if not force_rebuild and not _has_vector_changes(update):
        return ServingSyncSummary(
            update=update,
            serving_index_kind=serving_index_kind,
            serving_index_path=serving_index_path,
            rebuilt_serving_index=False,
            compression=None,
        )

    compression = _build_serving_index(
        exact_index_path=Path(exact_index_path),
        serving_index_path=serving_index_path,
        serving_index_kind=serving_index_kind,
        method=compression_method,
        pca_dimensions=pca_dimensions,
        top_k=top_k,
        sample_size=sample_size,
        evaluate=evaluate,
    )
    if record_report:
        append_report(
            compression,
            jsonl_path=jsonl_report_path,
            markdown_path=markdown_report_path,
            label=label,
        )
    return ServingSyncSummary(
        update=update,
        serving_index_kind=serving_index_kind,
        serving_index_path=serving_index_path,
        rebuilt_serving_index=True,
        compression=compression,
    )


def _has_vector_changes(summary: IndexBuildSummary) -> bool:
    return summary.embedded > 0 or summary.deleted > 0


def _build_serving_index(
    *,
    exact_index_path: Path,
    serving_index_path: Path,
    serving_index_kind: str,
    method: str,
    pca_dimensions: int | None,
    top_k: int,
    sample_size: int,
    evaluate: Callable[..., CompressionReport],
) -> CompressionReport:
    if serving_index_kind == "int8":
        return evaluate(
            input_path=exact_index_path,
            output_path=serving_index_path,
            method=method,
            pca_dimensions=pca_dimensions,
            top_k=top_k,
            sample_size=sample_size,
        )

    if method != "int8" or pca_dimensions is not None:
        raise ValueError("int8_mmap serving index requires compression_method='int8' without PCA")

    temporary_int8_path = _temporary_int8_path(serving_index_path)
    try:
        report = evaluate(
            input_path=exact_index_path,
            output_path=temporary_int8_path,
            method=method,
            pca_dimensions=None,
            top_k=top_k,
            sample_size=sample_size,
        )
        conversion = convert_int8_index_to_mmap(
            input_path=temporary_int8_path,
            output_path=serving_index_path,
            overwrite=True,
        )
        return replace(
            report,
            output_path=serving_index_path,
            output_bytes=conversion.output_bytes,
        )
    finally:
        _unlink_if_exists(temporary_int8_path)


def _temporary_int8_path(serving_index_path: Path) -> Path:
    return serving_index_path.with_name(f"{serving_index_path.name}.tmp.npz")


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume OAI updates for the exact index, then refresh the serving index if vectors changed."
    )
    parser.add_argument("--endpoint", default=OAI_ENDPOINT)
    parser.add_argument("--from-date", dest="from_date")
    parser.add_argument("--until-date", dest="until_date")
    parser.add_argument("--max-batches", dest="batch_limit", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--embedder", choices=("sentence-transformers", "hashing"), default="sentence-transformers")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--request-delay-seconds", type=float, default=3.0)
    parser.add_argument("--fetch-retries", type=int, default=DEFAULT_FETCH_RETRIES)
    parser.add_argument("--fetch-retry-delay-seconds", type=float, default=DEFAULT_RETRY_DELAY_SECONDS)
    parser.add_argument("--checkpoint-every-records", type=int, default=10_000)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--compression-method", choices=("int8", "pca-int8", "pca-float"), default="int8")
    parser.add_argument("--pca-dimensions", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=1_000)
    parser.add_argument("--label", default="daily-int8")
    parser.add_argument("--db-path", type=Path, default=Path("data/paper_recommender_1m.db"))
    parser.add_argument("--exact-index-path", type=Path, default=Path("data/vectors_1m.npz"))
    parser.add_argument("--serving-index-path", type=Path, default=Path("data/vectors_1m_int8.npz"))
    parser.add_argument("--serving-index-kind", choices=("int8", "int8_mmap"), default="int8")
    parser.add_argument("--jsonl-report", type=Path, default=Path("docs/evaluations/compression-runs.jsonl"))
    parser.add_argument("--markdown-report", type=Path, default=Path("docs/evaluations/compression-runs.md"))
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    summary = sync_serving_index(
        endpoint=args.endpoint,
        db_path=args.db_path,
        exact_index_path=args.exact_index_path,
        serving_index_path=args.serving_index_path,
        serving_index_kind=args.serving_index_kind,
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
        compression_method=args.compression_method,
        pca_dimensions=args.pca_dimensions,
        top_k=args.top_k,
        sample_size=args.sample_size,
        label=args.label,
        jsonl_report_path=args.jsonl_report,
        markdown_report_path=args.markdown_report,
        force_rebuild=args.force_rebuild,
        record_report=not args.no_record,
    )
    print(_format_summary(summary))


def _format_summary(summary: ServingSyncSummary) -> str:
    update = summary.update
    parts = [
        "Serving sync:",
        f"records={update.records_seen}",
        f"embedded={update.embedded}",
        f"deleted={update.deleted}",
        f"last_datestamp={update.last_datestamp}",
        f"serving_index_kind={summary.serving_index_kind}",
        f"serving_index_path={summary.serving_index_path}",
        f"rebuilt_serving_index={summary.rebuilt_serving_index}",
    ]
    if summary.compression is not None:
        parts.extend(
            [
                f"size_ratio={summary.compression.size_ratio:.4f}",
                f"recall@{summary.compression.recall.k}={summary.compression.recall.recall:.4f}",
                f"queries={summary.compression.recall.queries}",
            ]
        )
    return " ".join(parts)


if __name__ == "__main__":
    main()

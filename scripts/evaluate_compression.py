from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from paper_recommender.compressed_vector_store import (
    Int8VectorIndex,
    PcaFloatVectorIndex,
    PcaInt8VectorIndex,
    RecallResult,
    recall_at_k,
)
from paper_recommender.vector_store import ExactVectorIndex


@dataclass(frozen=True)
class CompressionReport:
    method: str
    input_path: Path
    output_path: Path
    pca_dimensions: int | None
    sample_size: int
    input_bytes: int
    output_bytes: int
    recall: RecallResult

    @property
    def size_ratio(self) -> float:
        if self.input_bytes == 0:
            return 0.0
        return self.output_bytes / self.input_bytes


def evaluate_compression(
    *,
    input_path: str | Path,
    output_path: str | Path,
    method: str,
    pca_dimensions: int | None,
    top_k: int,
    sample_size: int,
) -> CompressionReport:
    input_path = Path(input_path)
    output_path = Path(output_path)
    exact = ExactVectorIndex.load(input_path)
    compressed = _build_candidate_index(exact, method=method, pca_dimensions=pca_dimensions)
    compressed.save(output_path)

    query_vector_ids = exact.vector_ids[:sample_size].astype(int).tolist()
    recall = recall_at_k(exact, compressed, query_vector_ids=query_vector_ids, k=top_k)
    return CompressionReport(
        method=method,
        input_path=input_path,
        output_path=output_path,
        pca_dimensions=pca_dimensions,
        sample_size=sample_size,
        input_bytes=input_path.stat().st_size,
        output_bytes=output_path.stat().st_size,
        recall=recall,
    )


def append_report(
    report: CompressionReport,
    *,
    jsonl_path: str | Path,
    markdown_path: str | Path,
    label: str,
) -> None:
    jsonl_path = Path(jsonl_path)
    markdown_path = Path(markdown_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    row = {
        "timestamp": timestamp,
        "label": label,
        "method": report.method,
        "input_path": str(report.input_path),
        "output_path": str(report.output_path),
        "pca_dimensions": report.pca_dimensions,
        "top_k": report.recall.k,
        "sample_size": report.sample_size,
        "queries": report.recall.queries,
        "recall": report.recall.recall,
        "input_bytes": report.input_bytes,
        "output_bytes": report.output_bytes,
        "size_ratio": report.size_ratio,
    }
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")

    if not markdown_path.exists():
        markdown_path.write_text(
            "# Compression Evaluation Runs\n\n"
            "| Timestamp | Label | PCA dims | top-k | Queries | Recall | Input bytes | "
            "Output bytes | Size ratio |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
            encoding="utf-8",
        )
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"| {timestamp} | {label} | {report.pca_dimensions} | {report.recall.k} | "
            f"{report.recall.queries} | {report.recall.recall:.4f} | "
            f"{report.input_bytes} | {report.output_bytes} | {report.size_ratio:.4f} |\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate a PCA + int8 compressed vector index."
    )
    parser.add_argument("--input", dest="input_path", type=Path, default=Path("data/vectors.npz"))
    parser.add_argument(
        "--output",
        dest="output_path",
        type=Path,
        default=Path("data/vectors_pca_int8.npz"),
    )
    parser.add_argument(
        "--method",
        choices=("pca-int8", "pca-float", "int8"),
        default="pca-int8",
    )
    parser.add_argument("--pca-dimensions", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--label", default="manual")
    parser.add_argument(
        "--jsonl-report",
        type=Path,
        default=Path("docs/evaluations/compression-runs.jsonl"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("docs/evaluations/compression-runs.md"),
    )
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    report = evaluate_compression(
        input_path=args.input_path,
        output_path=args.output_path,
        method=args.method,
        pca_dimensions=args.pca_dimensions,
        top_k=args.top_k,
        sample_size=args.sample_size,
    )
    if not args.no_record:
        append_report(
            report,
            jsonl_path=args.jsonl_report,
            markdown_path=args.markdown_report,
            label=args.label,
        )
    print(
        "Compression report: "
        f"input_bytes={report.input_bytes}, "
        f"output_bytes={report.output_bytes}, "
        f"method={report.method}, "
        f"size_ratio={report.size_ratio:.4f}, "
        f"recall@{report.recall.k}={report.recall.recall:.4f}, "
        f"queries={report.recall.queries}"
    )


def _build_candidate_index(
    exact: ExactVectorIndex,
    *,
    method: str,
    pca_dimensions: int | None,
):
    if method == "pca-int8":
        if pca_dimensions is None:
            raise ValueError("pca_dimensions is required for pca-int8")
        return PcaInt8VectorIndex.from_exact_index(exact, pca_dimensions=pca_dimensions)
    if method == "pca-float":
        if pca_dimensions is None:
            raise ValueError("pca_dimensions is required for pca-float")
        return PcaFloatVectorIndex.from_exact_index(exact, pca_dimensions=pca_dimensions)
    if method == "int8":
        return Int8VectorIndex.from_exact_index(exact)
    raise ValueError(f"unknown compression method: {method}")


if __name__ == "__main__":
    main()

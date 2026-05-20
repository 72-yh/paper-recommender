from __future__ import annotations

import argparse
from pathlib import Path

from paper_recommender.index_builder import build_index_from_oai
from paper_recommender.oai_client import OAI_ENDPOINT
from paper_recommender.embedding import DEFAULT_MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local Paper Recommender index from arXiv OAI-PMH.")
    parser.add_argument("--endpoint", default=OAI_ENDPOINT)
    parser.add_argument("--from-date", dest="from_date")
    parser.add_argument("--until-date", dest="until_date")
    parser.add_argument("--max-batches", dest="batch_limit", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument(
        "--embedder",
        choices=("sentence-transformers", "hashing"),
        default="sentence-transformers",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--request-delay-seconds", type=float, default=0.0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every-batches", type=int)
    parser.add_argument("--checkpoint-every-records", type=int)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--target-vector-count", type=int)
    parser.add_argument("--db-path", type=Path, default=Path("data/paper_recommender.db"))
    parser.add_argument("--index-path", type=Path, default=Path("data/vectors.npz"))
    args = parser.parse_args()

    summary = build_index_from_oai(
        endpoint=args.endpoint,
        db_path=args.db_path,
        index_path=args.index_path,
        from_date=args.from_date,
        until_date=args.until_date,
        batch_limit=args.batch_limit,
        max_records=args.max_records,
        embedder_backend=args.embedder,
        model_name=args.model_name,
        device=args.device,
        dimensions=args.dimensions,
        reset=args.reset,
        resume=args.resume,
        request_delay_seconds=args.request_delay_seconds,
        checkpoint_every_batches=args.checkpoint_every_batches,
        checkpoint_every_records=args.checkpoint_every_records,
        embedding_batch_size=args.embedding_batch_size,
        target_vector_count=args.target_vector_count,
    )
    print(
        "Built index: "
        f"batches={summary.batches_seen}, "
        f"records={summary.records_seen}, "
        f"embedded={summary.embedded}, "
        f"deleted={summary.deleted}, "
        f"checkpoints={summary.checkpoints_written}, "
        f"last_datestamp={summary.last_datestamp}"
    )


if __name__ == "__main__":
    main()

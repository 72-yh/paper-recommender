from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from paper_recommender.embedding import DEFAULT_MODEL_NAME, embedding_text, make_embedder
from paper_recommender.oai_client import OAI_ENDPOINT, fetch_oai_batches
from paper_recommender.pipeline import apply_oai_record
from paper_recommender.storage import (
    connect_db,
    get_paper,
    init_db,
    list_active_papers_with_vectors,
    max_vector_id,
    set_paper_vector_id,
    set_pipeline_state,
)
from paper_recommender.vector_store import ExactVectorIndex


@dataclass(frozen=True)
class IndexBuildSummary:
    batches_seen: int
    records_seen: int
    inserted: int
    updated: int
    unchanged: int
    deleted: int
    embedded: int
    last_datestamp: str | None


def build_index_from_oai(
    *,
    endpoint: str = OAI_ENDPOINT,
    db_path: str | Path = "data/paper_recommender.db",
    index_path: str | Path = "data/vectors.npz",
    from_date: str | None = None,
    until_date: str | None = None,
    batch_limit: int | None = None,
    max_records: int | None = None,
    embedder=None,
    embedder_backend: str = "sentence-transformers",
    model_name: str = DEFAULT_MODEL_NAME,
    dimensions: int = 256,
    reset: bool = False,
    fetch_text: Callable[[str], str] | None = None,
    request_delay_seconds: float = 0.0,
) -> IndexBuildSummary:
    db_path = Path(db_path)
    index_path = Path(index_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        _unlink_if_exists(db_path)
        _unlink_if_exists(index_path)

    conn = connect_db(db_path)
    init_db(conn)
    embedder = embedder or make_embedder(
        embedder_backend,
        model_name=model_name,
        dimensions=dimensions,
    )
    items = _load_existing_items(conn, index_path, embedder.dimensions)
    next_vector_id = max(max_vector_id(conn), max(items, default=0)) + 1

    stats = {
        "batches_seen": 0,
        "records_seen": 0,
        "inserted": 0,
        "updated": 0,
        "unchanged": 0,
        "deleted": 0,
        "embedded": 0,
    }
    last_datestamp: str | None = None

    try:
        stop = False
        for batch in fetch_oai_batches(
            endpoint,
            from_date=from_date,
            until_date=until_date,
            batch_limit=batch_limit,
            fetch_text=fetch_text,
            request_delay_seconds=request_delay_seconds,
        ):
            stats["batches_seen"] += 1
            pending_embeddings: list[tuple[int, str]] = []
            for record in batch.records:
                if max_records is not None and stats["records_seen"] >= max_records:
                    stop = True
                    break

                before = get_paper(conn, record.arxiv_id)
                old_vector_id = before.vector_id if before is not None else None
                decision = apply_oai_record(conn, record)
                stats["records_seen"] += 1
                stats[decision] += 1
                last_datestamp = record.oai_datestamp

                if decision == "deleted":
                    if old_vector_id is not None:
                        items.pop(old_vector_id, None)
                    continue

                paper = get_paper(conn, record.arxiv_id)
                if paper is None:
                    continue

                needs_embedding = decision in {"inserted", "updated"}
                if paper.vector_id is not None and paper.vector_id not in items:
                    needs_embedding = True
                if paper.vector_id is None:
                    needs_embedding = True

                if needs_embedding:
                    vector_id = old_vector_id or paper.vector_id or next_vector_id
                    if vector_id == next_vector_id:
                        next_vector_id += 1
                    if old_vector_id is not None and old_vector_id != vector_id:
                        items.pop(old_vector_id, None)
                    set_paper_vector_id(conn, record.arxiv_id, vector_id)
                    pending_embeddings.append((vector_id, embedding_text(record)))
                    stats["embedded"] += 1

            if pending_embeddings:
                texts = [text for _, text in pending_embeddings]
                vectors = embedder.embed_texts(texts)
                for (vector_id, _), vector in zip(pending_embeddings, vectors, strict=True):
                    items[vector_id] = vector

            if stop:
                break

        ExactVectorIndex.from_items(items).save(index_path)
        if last_datestamp is not None:
            set_pipeline_state(conn, "last_successful_oai_datestamp", last_datestamp)
    finally:
        conn.close()

    return IndexBuildSummary(last_datestamp=last_datestamp, **stats)


def _load_existing_items(conn, index_path: Path, dimensions: int) -> dict[int, np.ndarray]:
    if not index_path.exists():
        return {}

    index = ExactVectorIndex.load(index_path)
    items: dict[int, np.ndarray] = {}
    for paper in list_active_papers_with_vectors(conn):
        if paper.vector_id is None:
            continue
        vector = index.get(paper.vector_id)
        if vector is not None and vector.shape == (dimensions,):
            items[paper.vector_id] = vector
    return items


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

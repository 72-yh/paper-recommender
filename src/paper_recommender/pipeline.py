from __future__ import annotations

import hashlib
import sqlite3

from paper_recommender.models import Paper
from paper_recommender.oai import OaiRecord
from paper_recommender.storage import get_paper, mark_deleted, upsert_paper


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def compute_content_hash(title: str, abstract: str, categories: tuple[str, ...]) -> str:
    payload = "\n".join(
        [
            _normalize_text(title),
            _normalize_text(abstract),
            " ".join(sorted(categories)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_oai_record(conn: sqlite3.Connection, record: OaiRecord) -> str:
    if record.deleted:
        mark_deleted(conn, record.arxiv_id, record.oai_datestamp)
        return "deleted"

    title = record.title or ""
    abstract = record.abstract or ""
    content_hash = compute_content_hash(title, abstract, record.categories)
    existing = get_paper(conn, record.arxiv_id)
    if existing and existing.content_hash == content_hash:
        upsert_paper(
            conn,
            Paper(
                arxiv_id=record.arxiv_id,
                vector_id=existing.vector_id,
                active=True,
                oai_datestamp=record.oai_datestamp,
                published_date=record.published_date,
                updated_date=record.updated_date,
                primary_category=record.categories[0] if record.categories else "",
                categories=record.categories,
                content_hash=content_hash,
            ),
        )
        return "unchanged"

    paper = Paper(
        arxiv_id=record.arxiv_id,
        vector_id=None,
        active=True,
        oai_datestamp=record.oai_datestamp,
        published_date=record.published_date,
        updated_date=record.updated_date,
        primary_category=record.categories[0] if record.categories else "",
        categories=record.categories,
        content_hash=content_hash,
    )
    upsert_paper(conn, paper)
    return "inserted" if existing is None else "updated"

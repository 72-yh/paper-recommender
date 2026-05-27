from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from paper_recommender.models import Paper


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    vector_id INTEGER UNIQUE,
    active INTEGER NOT NULL,
    oai_datestamp TEXT NOT NULL,
    published_date TEXT,
    updated_date TEXT,
    primary_category TEXT NOT NULL,
    categories TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pipeline_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_deletes (
    arxiv_id TEXT NOT NULL,
    vector_id INTEGER,
    deleted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS preview_cache (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    snippet TEXT NOT NULL,
    cached_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CATEGORY_LOOKUP_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_categories (
    arxiv_id TEXT NOT NULL,
    category TEXT NOT NULL,
    vector_id INTEGER NOT NULL,
    published_date TEXT,
    PRIMARY KEY (arxiv_id, category),
    FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_paper_categories_lookup
ON paper_categories(category, published_date, vector_id);

CREATE INDEX IF NOT EXISTS idx_papers_active_published_vector
ON papers(published_date, vector_id)
WHERE active = 1 AND vector_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_papers_status_count
ON papers(active, vector_id);
"""

CATEGORY_LOOKUP_BACKFILL_STATE_KEY = "paper_categories_backfilled_v1"


def connect_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _ensure_category_lookup_backfilled(conn)


def _encode_categories(categories: tuple[str, ...]) -> str:
    return " ".join(categories)


def _decode_categories(value: str) -> tuple[str, ...]:
    return tuple(part for part in value.split(" ") if part)


def _row_to_paper(row: sqlite3.Row | None) -> Paper | None:
    if row is None:
        return None
    return Paper(
        arxiv_id=row["arxiv_id"],
        vector_id=row["vector_id"],
        active=bool(row["active"]),
        oai_datestamp=row["oai_datestamp"],
        published_date=row["published_date"],
        updated_date=row["updated_date"],
        primary_category=row["primary_category"],
        categories=_decode_categories(row["categories"]),
        content_hash=row["content_hash"],
    )


def upsert_paper(conn: sqlite3.Connection, paper: Paper, *, commit: bool = True) -> None:
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, vector_id, active, oai_datestamp, published_date, updated_date,
            primary_category, categories, content_hash, last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            vector_id = excluded.vector_id,
            active = excluded.active,
            oai_datestamp = excluded.oai_datestamp,
            published_date = excluded.published_date,
            updated_date = excluded.updated_date,
            primary_category = excluded.primary_category,
            categories = excluded.categories,
            content_hash = excluded.content_hash,
            last_seen_at = CURRENT_TIMESTAMP
        """,
        (
            paper.arxiv_id,
            paper.vector_id,
            int(paper.active),
            paper.oai_datestamp,
            paper.published_date,
            paper.updated_date,
            paper.primary_category,
            _encode_categories(paper.categories),
            paper.content_hash,
        ),
    )
    _sync_paper_categories(conn, paper.arxiv_id)
    if commit:
        conn.commit()


def get_paper(conn: sqlite3.Connection, arxiv_id: str) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
    return _row_to_paper(row)


def get_paper_by_vector_id(conn: sqlite3.Connection, vector_id: int) -> Paper | None:
    row = conn.execute("SELECT * FROM papers WHERE vector_id = ?", (vector_id,)).fetchone()
    return _row_to_paper(row)


def list_active_papers_with_vectors(conn: sqlite3.Connection) -> list[Paper]:
    rows = conn.execute(
        """
        SELECT *
        FROM papers
        WHERE active = 1 AND vector_id IS NOT NULL
        ORDER BY vector_id
        """
    ).fetchall()
    return [paper for row in rows if (paper := _row_to_paper(row)) is not None]


def list_active_category_counts(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    _ensure_category_lookup_backfilled(conn)
    rows = conn.execute(
        """
        SELECT category, COUNT(*) AS count
        FROM paper_categories
        GROUP BY category
        ORDER BY category
        """
    ).fetchall()
    return [(row["category"], int(row["count"])) for row in rows]


def list_filtered_vector_ids(
    conn: sqlite3.Connection,
    *,
    categories: tuple[str, ...],
    date_from: str | None = None,
    date_to: str | None = None,
    exclude_vector_id: int | None = None,
) -> list[int] | None:
    if not categories and date_from is None and date_to is None:
        return None

    _ensure_category_lookup_backfilled(conn)
    if categories:
        return _list_filtered_category_vector_ids(
            conn,
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            exclude_vector_id=exclude_vector_id,
        )

    where = ["active = 1", "vector_id IS NOT NULL"]
    params: list[object] = []
    if exclude_vector_id is not None:
        where.append("vector_id != ?")
        params.append(exclude_vector_id)
    if date_from is not None or date_to is not None:
        where.append("published_date IS NOT NULL")
    if date_from is not None:
        where.append("published_date >= ?")
        params.append(date_from)
    if date_to is not None:
        where.append("published_date <= ?")
        params.append(date_to)

    rows = conn.execute(
        f"""
        SELECT vector_id
        FROM papers
        WHERE {' AND '.join(where)}
        ORDER BY vector_id
        """,
        params,
    ).fetchall()
    return [int(row["vector_id"]) for row in rows]


def max_vector_id(conn: sqlite3.Connection) -> int:
    row: Any = conn.execute("SELECT COALESCE(MAX(vector_id), 0) AS value FROM papers").fetchone()
    return int(row["value"])


def set_paper_vector_id(
    conn: sqlite3.Connection,
    arxiv_id: str,
    vector_id: int,
    *,
    commit: bool = True,
) -> None:
    conn.execute(
        """
        UPDATE papers
        SET vector_id = ?, last_seen_at = CURRENT_TIMESTAMP
        WHERE arxiv_id = ?
        """,
        (vector_id, arxiv_id),
    )
    _sync_paper_categories(conn, arxiv_id)
    if commit:
        conn.commit()


def update_oai_datestamp(
    conn: sqlite3.Connection,
    arxiv_id: str,
    oai_datestamp: str,
    *,
    commit: bool = True,
) -> None:
    conn.execute(
        """
        UPDATE papers
        SET oai_datestamp = ?, last_seen_at = CURRENT_TIMESTAMP
        WHERE arxiv_id = ?
        """,
        (oai_datestamp, arxiv_id),
    )
    if commit:
        conn.commit()


def mark_deleted(
    conn: sqlite3.Connection,
    arxiv_id: str,
    oai_datestamp: str,
    *,
    commit: bool = True,
) -> None:
    existing = get_paper(conn, arxiv_id)
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, vector_id, active, oai_datestamp, published_date, updated_date,
            primary_category, categories, content_hash, last_seen_at
        )
        VALUES (?, NULL, 0, ?, NULL, NULL, '', '', '', CURRENT_TIMESTAMP)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            vector_id = NULL,
            active = 0,
            oai_datestamp = excluded.oai_datestamp,
            last_seen_at = CURRENT_TIMESTAMP
        """,
        (arxiv_id, oai_datestamp),
    )
    if existing and existing.vector_id is not None:
        conn.execute(
            "INSERT INTO index_deletes (arxiv_id, vector_id) VALUES (?, ?)",
            (arxiv_id, existing.vector_id),
        )
    conn.execute("DELETE FROM paper_categories WHERE arxiv_id = ?", (arxiv_id,))
    if commit:
        conn.commit()


def set_pipeline_state(
    conn: sqlite3.Connection,
    key: str,
    value: str,
    *,
    commit: bool = True,
) -> None:
    conn.execute(
        """
        INSERT INTO pipeline_state (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    if commit:
        conn.commit()


def get_pipeline_state(conn: sqlite3.Connection, key: str) -> str | None:
    row: Any = conn.execute("SELECT value FROM pipeline_state WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def _ensure_category_lookup_backfilled(conn: sqlite3.Connection) -> None:
    conn.executescript(CATEGORY_LOOKUP_SCHEMA)
    if get_pipeline_state(conn, CATEGORY_LOOKUP_BACKFILL_STATE_KEY) == "1":
        return

    _rebuild_category_lookup(conn)
    set_pipeline_state(conn, CATEGORY_LOOKUP_BACKFILL_STATE_KEY, "1", commit=False)
    conn.commit()


def _rebuild_category_lookup(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM paper_categories")
    rows = conn.execute(
        """
        SELECT arxiv_id, vector_id, published_date, categories
        FROM papers
        WHERE active = 1 AND vector_id IS NOT NULL
        """
    )
    for row in rows:
        _insert_category_lookup_rows(
            conn,
            arxiv_id=row["arxiv_id"],
            vector_id=int(row["vector_id"]),
            published_date=row["published_date"],
            categories=_decode_categories(row["categories"]),
        )


def _sync_paper_categories(conn: sqlite3.Connection, arxiv_id: str) -> None:
    conn.execute("DELETE FROM paper_categories WHERE arxiv_id = ?", (arxiv_id,))
    row = conn.execute(
        """
        SELECT arxiv_id, vector_id, active, published_date, categories
        FROM papers
        WHERE arxiv_id = ?
        """,
        (arxiv_id,),
    ).fetchone()
    if row is None or not row["active"] or row["vector_id"] is None:
        return

    _insert_category_lookup_rows(
        conn,
        arxiv_id=row["arxiv_id"],
        vector_id=int(row["vector_id"]),
        published_date=row["published_date"],
        categories=_decode_categories(row["categories"]),
    )


def _insert_category_lookup_rows(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str,
    vector_id: int,
    published_date: str | None,
    categories: tuple[str, ...],
) -> None:
    unique_categories = tuple(dict.fromkeys(categories))
    if not unique_categories:
        return
    conn.executemany(
        """
        INSERT INTO paper_categories (arxiv_id, category, vector_id, published_date)
        VALUES (?, ?, ?, ?)
        """,
        [(arxiv_id, category, vector_id, published_date) for category in unique_categories],
    )


def _list_filtered_category_vector_ids(
    conn: sqlite3.Connection,
    *,
    categories: tuple[str, ...],
    date_from: str | None,
    date_to: str | None,
    exclude_vector_id: int | None,
) -> list[int]:
    placeholders = ", ".join("?" for _ in categories)
    where = [f"category IN ({placeholders})"]
    params: list[object] = list(categories)
    if exclude_vector_id is not None:
        where.append("vector_id != ?")
        params.append(exclude_vector_id)
    if date_from is not None or date_to is not None:
        where.append("published_date IS NOT NULL")
    if date_from is not None:
        where.append("published_date >= ?")
        params.append(date_from)
    if date_to is not None:
        where.append("published_date <= ?")
        params.append(date_to)

    rows = conn.execute(
        f"""
        SELECT DISTINCT vector_id
        FROM paper_categories
        WHERE {' AND '.join(where)}
        ORDER BY vector_id
        """,
        params,
    ).fetchall()
    return [int(row["vector_id"]) for row in rows]

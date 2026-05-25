from paper_recommender.models import Paper
from paper_recommender.storage import (
    connect_db,
    get_paper,
    get_paper_by_vector_id,
    get_pipeline_state,
    init_db,
    list_active_category_counts,
    list_filtered_vector_ids,
    mark_deleted,
    set_paper_vector_id,
    set_pipeline_state,
    update_oai_datestamp,
    upsert_paper,
)


def test_upsert_and_get_paper() -> None:
    conn = connect_db(":memory:")
    init_db(conn)

    paper = Paper(
        arxiv_id="1706.03762",
        vector_id=1,
        active=True,
        oai_datestamp="2024-01-02",
        published_date="2017-06-12",
        updated_date="2023-08-02",
        primary_category="cs.CL",
        categories=("cs.CL", "cs.LG"),
        content_hash="hash-a",
    )
    upsert_paper(conn, paper)

    stored = get_paper(conn, "1706.03762")

    assert stored == paper
    assert get_paper_by_vector_id(conn, 1) == paper


def test_upsert_paper_maintains_indexed_category_lookup() -> None:
    conn = connect_db(":memory:")
    init_db(conn)

    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date="2017-06-12",
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL", "cs.LG"),
            content_hash="hash-a",
        ),
    )

    rows = conn.execute(
        """
        SELECT category, vector_id, published_date
        FROM paper_categories
        ORDER BY category
        """
    ).fetchall()

    assert [tuple(row) for row in rows] == [
        ("cs.CL", 1, "2017-06-12"),
        ("cs.LG", 1, "2017-06-12"),
    ]
    assert list_active_category_counts(conn) == [("cs.CL", 1), ("cs.LG", 1)]


def test_set_paper_vector_id_adds_deferred_category_lookup_rows() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=None,
            active=True,
            oai_datestamp="2024-01-02",
            published_date="2017-06-12",
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL", "cs.LG"),
            content_hash="hash-a",
        ),
    )

    assert conn.execute("SELECT COUNT(*) FROM paper_categories").fetchone()[0] == 0

    set_paper_vector_id(conn, "1706.03762", 1)

    assert list_filtered_vector_ids(conn, categories=("cs.LG",)) == [1]


def test_upsert_paper_replaces_stale_category_lookup_rows() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date=None,
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL", "cs.LG"),
            content_hash="hash-a",
        ),
    )

    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-03",
            published_date=None,
            updated_date=None,
            primary_category="stat.ML",
            categories=("stat.ML",),
            content_hash="hash-b",
        ),
    )

    assert list_active_category_counts(conn) == [("stat.ML", 1)]
    assert list_filtered_vector_ids(conn, categories=("cs.CL",)) == []


def test_mark_deleted_removes_category_lookup_rows() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date=None,
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="hash-a",
        ),
    )

    mark_deleted(conn, "1706.03762", "2024-01-03")

    assert list_active_category_counts(conn) == []
    assert list_filtered_vector_ids(conn, categories=("cs.CL",)) == []


def test_category_lookup_backfills_existing_database() -> None:
    conn = connect_db(":memory:")
    conn.executescript(
        """
        CREATE TABLE papers (
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
        CREATE TABLE pipeline_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, vector_id, active, oai_datestamp, published_date,
            updated_date, primary_category, categories, content_hash
        )
        VALUES
            ('1706.03762', 1, 1, '2024-01-02', '2017-06-12', NULL, 'cs.CL', 'cs.CL cs.LG', 'hash-a'),
            ('1111.11111', 2, 1, '2024-01-02', '2020-01-01', NULL, 'stat.ML', 'stat.ML', 'hash-b'),
            ('2222.22222', NULL, 1, '2024-01-02', '2020-01-01', NULL, 'cs.CL', 'cs.CL', 'hash-c'),
            ('3333.33333', 3, 0, '2024-01-02', '2020-01-01', NULL, 'cs.CL', 'cs.CL', 'hash-d')
        """
    )

    assert list_active_category_counts(conn) == [("cs.CL", 1), ("cs.LG", 1), ("stat.ML", 1)]
    assert list_filtered_vector_ids(conn, categories=("cs.LG",)) == [1]
    assert conn.execute("SELECT COUNT(*) FROM paper_categories").fetchone()[0] == 3


def test_upsert_paper_can_defer_commit(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    conn = connect_db(db_path)
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date="2017-06-12",
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="hash-a",
        ),
        commit=False,
    )

    other = connect_db(db_path)
    try:
        assert get_paper(other, "1706.03762") is None
        conn.commit()
        assert get_paper(other, "1706.03762") is not None
    finally:
        other.close()


def test_mark_deleted_deactivates_paper_and_records_tombstone() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date=None,
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="hash-a",
        ),
    )

    mark_deleted(conn, "1706.03762", "2024-01-03")

    stored = get_paper(conn, "1706.03762")
    tombstones = conn.execute("SELECT arxiv_id, vector_id FROM index_deletes").fetchall()

    assert stored is not None
    assert stored.active is False
    assert stored.vector_id is None
    assert get_paper_by_vector_id(conn, 1) is None
    assert [tuple(row) for row in tombstones] == [("1706.03762", 1)]


def test_mark_deleted_creates_inactive_missing_record_without_tombstone() -> None:
    conn = connect_db(":memory:")
    init_db(conn)

    mark_deleted(conn, "1706.03762", "2024-01-03")

    stored = get_paper(conn, "1706.03762")
    tombstones = conn.execute("SELECT arxiv_id, vector_id FROM index_deletes").fetchall()

    assert stored is not None
    assert stored.active is False
    assert stored.vector_id is None
    assert [tuple(row) for row in tombstones] == []


def test_mark_deleted_without_vector_id_does_not_record_tombstone() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=None,
            active=True,
            oai_datestamp="2024-01-02",
            published_date=None,
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="hash-a",
        ),
    )

    mark_deleted(conn, "1706.03762", "2024-01-03")

    stored = get_paper(conn, "1706.03762")
    tombstones = conn.execute("SELECT arxiv_id, vector_id FROM index_deletes").fetchall()

    assert stored is not None
    assert stored.active is False
    assert stored.vector_id is None
    assert [tuple(row) for row in tombstones] == []


def test_update_oai_datestamp_without_reembedding() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date=None,
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="hash-a",
        ),
    )

    update_oai_datestamp(conn, "1706.03762", "2024-01-04")

    stored = get_paper(conn, "1706.03762")
    assert stored is not None
    assert stored.oai_datestamp == "2024-01-04"
    assert stored.content_hash == "hash-a"


def test_pipeline_state_get_set() -> None:
    conn = connect_db(":memory:")
    init_db(conn)

    assert get_pipeline_state(conn, "last_successful_oai_datestamp") is None

    set_pipeline_state(conn, "last_successful_oai_datestamp", "2024-01-02")
    set_pipeline_state(conn, "last_successful_oai_datestamp", "2024-01-03")

    assert get_pipeline_state(conn, "last_successful_oai_datestamp") == "2024-01-03"

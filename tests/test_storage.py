from paper_recommender.models import Paper
from paper_recommender.storage import (
    connect_db,
    get_paper,
    get_paper_by_vector_id,
    get_pipeline_state,
    init_db,
    mark_deleted,
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

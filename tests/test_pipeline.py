from paper_recommender.models import Paper
from paper_recommender.oai import OaiRecord
from paper_recommender.pipeline import apply_oai_record, compute_content_hash
from paper_recommender.storage import connect_db, get_paper, init_db, upsert_paper


def test_content_hash_ignores_whitespace_noise() -> None:
    first = compute_content_hash("A  Title", "Line one\nline two", ("cs.CL", "cs.LG"))
    second = compute_content_hash("A Title", "Line one line two", ("cs.LG", "cs.CL"))

    assert first == second


def test_apply_new_record_inserts_paper_without_vector() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    record = OaiRecord(
        arxiv_id="1706.03762",
        oai_datestamp="2024-01-02",
        deleted=False,
        title="Attention",
        abstract="Abstract",
        categories=("cs.CL",),
        published_date="2017-06-12",
        updated_date="2023-08-02",
    )

    decision = apply_oai_record(conn, record)

    stored = get_paper(conn, "1706.03762")
    assert decision == "inserted"
    assert stored is not None
    assert stored.vector_id is None
    assert stored.active is True


def test_apply_unchanged_record_updates_datestamp_only() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    content_hash = compute_content_hash("Attention", "Abstract", ("cs.CL",))
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date="2017-06-12",
            updated_date="2023-08-02",
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash=content_hash,
        ),
    )

    decision = apply_oai_record(
        conn,
        OaiRecord(
            arxiv_id="1706.03762",
            oai_datestamp="2024-01-04",
            deleted=False,
            title="Attention",
            abstract="Abstract",
            categories=("cs.CL",),
            published_date="2017-06-12",
            updated_date="2023-08-02",
        ),
    )

    stored = get_paper(conn, "1706.03762")
    assert decision == "unchanged"
    assert stored is not None
    assert stored.vector_id == 1
    assert stored.oai_datestamp == "2024-01-04"


def test_apply_unchanged_record_refreshes_metadata_without_clearing_vector() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    content_hash = compute_content_hash("Attention", "Abstract", ("cs.CL", "cs.LG"))
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date="2017-06-12",
            updated_date="2023-08-02",
            primary_category="cs.CL",
            categories=("cs.CL", "cs.LG"),
            content_hash=content_hash,
        ),
    )

    decision = apply_oai_record(
        conn,
        OaiRecord(
            arxiv_id="1706.03762",
            oai_datestamp="2024-01-04",
            deleted=False,
            title="Attention",
            abstract="Abstract",
            categories=("cs.LG", "cs.CL"),
            published_date="2017-06-13",
            updated_date="2024-01-01",
        ),
    )

    stored = get_paper(conn, "1706.03762")
    assert decision == "unchanged"
    assert stored is not None
    assert stored.vector_id == 1
    assert stored.oai_datestamp == "2024-01-04"
    assert stored.published_date == "2017-06-13"
    assert stored.updated_date == "2024-01-01"
    assert stored.primary_category == "cs.LG"
    assert stored.categories == ("cs.LG", "cs.CL")


def test_apply_unchanged_record_reactivates_existing_paper_without_clearing_vector() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    content_hash = compute_content_hash("Attention", "Abstract", ("cs.CL",))
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=False,
            oai_datestamp="2024-01-02",
            published_date="2017-06-12",
            updated_date="2023-08-02",
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash=content_hash,
        ),
    )

    decision = apply_oai_record(
        conn,
        OaiRecord(
            arxiv_id="1706.03762",
            oai_datestamp="2024-01-04",
            deleted=False,
            title="Attention",
            abstract="Abstract",
            categories=("cs.CL",),
            published_date="2017-06-13",
            updated_date="2024-01-01",
        ),
    )

    stored = get_paper(conn, "1706.03762")
    assert decision == "unchanged"
    assert stored is not None
    assert stored.active is True
    assert stored.vector_id == 1
    assert stored.published_date == "2017-06-13"
    assert stored.updated_date == "2024-01-01"


def test_apply_changed_record_clears_vector_for_reembedding() -> None:
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
            content_hash="old-hash",
        ),
    )

    decision = apply_oai_record(
        conn,
        OaiRecord(
            arxiv_id="1706.03762",
            oai_datestamp="2024-01-05",
            deleted=False,
            title="New title",
            abstract="New abstract",
            categories=("cs.CL", "cs.LG"),
            published_date=None,
            updated_date=None,
        ),
    )

    stored = get_paper(conn, "1706.03762")
    assert decision == "updated"
    assert stored is not None
    assert stored.vector_id is None
    assert stored.categories == ("cs.CL", "cs.LG")


def test_apply_deleted_record_marks_inactive_and_clears_vector() -> None:
    conn = connect_db(":memory:")
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="9999.00001",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-02",
            published_date=None,
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="old-hash",
        ),
    )

    decision = apply_oai_record(
        conn,
        OaiRecord(
            arxiv_id="9999.00001",
            oai_datestamp="2024-01-03",
            deleted=True,
            title=None,
            abstract=None,
            categories=(),
            published_date=None,
            updated_date=None,
        ),
    )

    stored = get_paper(conn, "9999.00001")
    assert decision == "deleted"
    assert stored is not None
    assert stored.active is False
    assert stored.vector_id is None

from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from paper_recommender.index_builder import build_index_from_oai
from paper_recommender.embedding import HashingTextEmbedder
from paper_recommender.models import Paper
from paper_recommender.recommender import recommend
from paper_recommender.storage import (
    connect_db,
    get_paper,
    get_pipeline_state,
    init_db,
    set_pipeline_state,
    upsert_paper,
)
from paper_recommender.vector_store import ExactVectorIndex


OAI_XML = """
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:1706.03762</identifier>
        <datestamp>2024-01-01</datestamp>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>1706.03762</id>
          <created>2017-06-12</created>
          <title>Attention Is All You Need</title>
          <abstract>Transformer attention models for sequence transduction.</abstract>
          <categories>cs.CL cs.LG</categories>
        </arXiv>
      </metadata>
    </record>
    <record>
      <header>
        <identifier>oai:arXiv.org:1810.04805</identifier>
        <datestamp>2024-01-01</datestamp>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>1810.04805</id>
          <created>2018-10-11</created>
          <title>BERT Pre-training of Deep Bidirectional Transformers</title>
          <abstract>Language representation models use transformer attention.</abstract>
          <categories>cs.CL</categories>
        </arXiv>
      </metadata>
    </record>
    <record>
      <header>
        <identifier>oai:arXiv.org:1406.2661</identifier>
        <datestamp>2024-01-01</datestamp>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>1406.2661</id>
          <created>2014-06-10</created>
          <title>Generative Adversarial Networks</title>
          <abstract>Generative models are trained with an adversarial process.</abstract>
          <categories>stat.ML cs.LG</categories>
        </arXiv>
      </metadata>
    </record>
    <record>
      <header status="deleted">
        <identifier>oai:arXiv.org:9999.00001</identifier>
        <datestamp>2024-01-02</datestamp>
      </header>
    </record>
  </ListRecords>
</OAI-PMH>
"""


def _single_record_xml(
    arxiv_id: str,
    datestamp: str,
    *,
    title: str = "Checkpoint Paper",
    token: str | None = None,
) -> str:
    token_xml = f"<resumptionToken>{token}</resumptionToken>" if token else ""
    return f"""
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:{arxiv_id}</identifier>
        <datestamp>{datestamp}</datestamp>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>{arxiv_id}</id>
          <created>{datestamp}</created>
          <title>{title}</title>
          <abstract>Checkpointable indexing for a larger corpus.</abstract>
          <categories>cs.IR cs.DL</categories>
        </arXiv>
      </metadata>
    </record>
    {token_xml}
  </ListRecords>
</OAI-PMH>
"""


def _many_records_xml(count: int) -> str:
    records = []
    for index in range(1, count + 1):
        arxiv_id = f"2401.{index:05d}"
        records.append(
            f"""
    <record>
      <header>
        <identifier>oai:arXiv.org:{arxiv_id}</identifier>
        <datestamp>2024-01-{index:02d}</datestamp>
      </header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>{arxiv_id}</id>
          <created>2024-01-{index:02d}</created>
          <title>Large batch paper {index}</title>
          <abstract>Large OAI batch indexing should flush embedding chunks.</abstract>
          <categories>cs.IR cs.DL</categories>
        </arXiv>
      </metadata>
    </record>
"""
        )
    return f"""
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    {"".join(records)}
  </ListRecords>
</OAI-PMH>
"""


def test_build_index_from_oai_writes_db_vectors_and_recommendations(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"

    summary = build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=db_path,
        index_path=index_path,
        from_date="2024-01-01",
        embedder=HashingTextEmbedder(dimensions=256),
        fetch_text=lambda _url: OAI_XML,
    )

    conn = connect_db(db_path)
    index = ExactVectorIndex.load(index_path)
    query = get_paper(conn, "1706.03762")
    deleted = get_paper(conn, "9999.00001")
    results = recommend(conn, index, "1706.03762", top_k=2)

    assert summary.records_seen == 4
    assert summary.embedded == 3
    assert summary.deleted == 1
    assert query is not None
    assert query.vector_id == 1
    assert index.get(query.vector_id) is not None
    assert deleted is not None
    assert deleted.active is False
    assert [result.arxiv_id for result in results] == ["1810.04805", "1406.2661"]


def test_build_index_from_oai_rebuilds_without_stale_updated_vectors(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=db_path,
        index_path=index_path,
        from_date="2024-01-01",
        embedder=HashingTextEmbedder(dimensions=256),
        fetch_text=lambda _url: OAI_XML,
    )

    updated_xml = OAI_XML.replace(
        "Transformer attention models for sequence transduction.",
        "Quantum circuits and Hamiltonian simulation.",
    )
    build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=db_path,
        index_path=index_path,
        from_date="2024-01-02",
        embedder=HashingTextEmbedder(dimensions=256),
        fetch_text=lambda _url: updated_xml,
    )

    conn = connect_db(db_path)
    index = ExactVectorIndex.load(index_path)
    active_vector_ids = [
        row["vector_id"]
        for row in conn.execute(
            "SELECT vector_id FROM papers WHERE active = 1 AND vector_id IS NOT NULL"
        )
    ]

    assert sorted(index.vector_ids.tolist()) == sorted(active_vector_ids)
    assert len(index.vector_ids) == 3
    assert all(np.linalg.norm(index.get(vector_id)) > 0 for vector_id in active_vector_ids)


def test_build_index_from_oai_drops_existing_vectors_with_wrong_dimensions(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    conn = connect_db(db_path)
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="0000.00001",
            vector_id=99,
            active=True,
            oai_datestamp="2023-01-01",
            published_date="2023-01-01",
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="old",
        ),
    )
    conn.close()
    ExactVectorIndex.from_items({99: np.array([1.0, 0.0, 0.0], dtype=np.float32)}).save(
        index_path
    )

    build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=db_path,
        index_path=index_path,
        from_date="2024-01-01",
        embedder=HashingTextEmbedder(dimensions=16),
        fetch_text=lambda _url: OAI_XML,
    )

    index = ExactVectorIndex.load(index_path)

    assert 99 not in index.vector_ids.tolist()
    assert index.vectors.shape == (3, 16)


def test_build_index_from_oai_reset_discards_existing_rows(tmp_path) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    conn = connect_db(db_path)
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="1111.11111",
            vector_id=1,
            active=True,
            oai_datestamp="2023-01-01",
            published_date="2023-01-01",
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="old",
        ),
    )
    conn.close()
    ExactVectorIndex.from_items({1: np.array([1.0, 0.0, 0.0], dtype=np.float32)}).save(index_path)

    build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=db_path,
        index_path=index_path,
        from_date="2024-01-01",
        embedder=HashingTextEmbedder(dimensions=16),
        reset=True,
        fetch_text=lambda _url: OAI_XML,
    )

    conn = connect_db(db_path)
    index = ExactVectorIndex.load(index_path)

    assert get_paper(conn, "1111.11111") is None
    assert get_paper(conn, "1706.03762") is not None
    assert index.vectors.shape == (3, 16)


def test_build_index_from_oai_batches_embeddings(tmp_path) -> None:
    class RecordingEmbedder:
        dimensions = 4

        def __init__(self) -> None:
            self.text_batches: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> np.ndarray:
            self.text_batches.append(texts)
            return np.eye(len(texts), self.dimensions, dtype=np.float32)

    embedder = RecordingEmbedder()

    build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=tmp_path / "papers.db",
        index_path=tmp_path / "vectors.npz",
        from_date="2024-01-01",
        embedder=embedder,
        fetch_text=lambda _url: OAI_XML,
    )

    assert len(embedder.text_batches) == 1
    assert len(embedder.text_batches[0]) == 3
    assert embedder.text_batches[0][0].startswith("Attention Is All You Need\n")


def test_build_index_from_oai_checkpoints_completed_batches_before_later_failure(
    tmp_path,
) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    responses = [
        _single_record_xml("2401.00001", "2024-01-01", token="next-batch"),
    ]

    def fetch_text(_url: str) -> str:
        if not responses:
            raise RuntimeError("simulated later OAI failure")
        return responses.pop(0)

    with pytest.raises(RuntimeError, match="simulated later OAI failure"):
        build_index_from_oai(
            endpoint="https://example.test/oai",
            db_path=db_path,
            index_path=index_path,
            from_date="2024-01-01",
            embedder=HashingTextEmbedder(dimensions=16),
            fetch_text=fetch_text,
            checkpoint_every_batches=1,
        )

    conn = connect_db(db_path)
    index = ExactVectorIndex.load(index_path)

    assert index.vector_ids.tolist() == [1]
    assert get_pipeline_state(conn, "last_successful_oai_datestamp") == "2024-01-01"


def test_build_index_from_oai_resume_uses_saved_datestamp_when_from_date_is_missing(
    tmp_path,
) -> None:
    db_path = tmp_path / "papers.db"
    conn = connect_db(db_path)
    init_db(conn)
    set_pipeline_state(conn, "last_successful_oai_datestamp", "2024-02-03")
    conn.close()
    fetched_urls: list[str] = []

    def fetch_text(url: str) -> str:
        fetched_urls.append(url)
        return _single_record_xml("2402.00001", "2024-02-04")

    build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=db_path,
        index_path=tmp_path / "vectors.npz",
        resume=True,
        embedder=HashingTextEmbedder(dimensions=16),
        fetch_text=fetch_text,
    )

    query = parse_qs(urlparse(fetched_urls[0]).query)

    assert query["from"] == ["2024-02-03"]


def test_build_index_from_oai_skips_fetch_when_target_vector_count_already_exists(
    tmp_path,
) -> None:
    db_path = tmp_path / "papers.db"
    index_path = tmp_path / "vectors.npz"
    conn = connect_db(db_path)
    init_db(conn)
    upsert_paper(
        conn,
        Paper(
            arxiv_id="2401.00001",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-01",
            published_date="2024-01-01",
            updated_date=None,
            primary_category="cs.IR",
            categories=("cs.IR",),
            content_hash="existing",
        ),
    )
    conn.close()
    ExactVectorIndex.from_items({1: np.array([1.0, 0.0], dtype=np.float32)}).save(index_path)
    fetched_urls: list[str] = []

    summary = build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=db_path,
        index_path=index_path,
        target_vector_count=1,
        embedder=HashingTextEmbedder(dimensions=2),
        fetch_text=lambda url: fetched_urls.append(url) or OAI_XML,
    )

    assert fetched_urls == []
    assert summary.records_seen == 0
    assert ExactVectorIndex.load(index_path).vector_ids.tolist() == [1]


def test_build_index_from_oai_flushes_embedding_chunks_inside_large_oai_batch(
    tmp_path,
) -> None:
    class RecordingEmbedder:
        dimensions = 4

        def __init__(self) -> None:
            self.text_batches: list[list[str]] = []

        def embed_texts(self, texts: list[str]) -> np.ndarray:
            self.text_batches.append(texts)
            return np.eye(len(texts), self.dimensions, dtype=np.float32)

    embedder = RecordingEmbedder()

    summary = build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=tmp_path / "papers.db",
        index_path=tmp_path / "vectors.npz",
        embedder=embedder,
        fetch_text=lambda _url: _many_records_xml(5),
        embedding_batch_size=2,
        checkpoint_every_records=2,
    )

    conn = connect_db(tmp_path / "papers.db")

    assert [len(batch) for batch in embedder.text_batches] == [2, 2, 1]
    assert summary.records_seen == 5
    assert summary.checkpoints_written == 2
    assert get_pipeline_state(conn, "last_successful_oai_datestamp") == "2024-01-05"
    assert ExactVectorIndex.load(tmp_path / "vectors.npz").vector_ids.tolist() == [1, 2, 3, 4, 5]


def test_build_index_from_oai_target_vector_count_stops_inside_large_oai_batch(
    tmp_path,
) -> None:
    summary = build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=tmp_path / "papers.db",
        index_path=tmp_path / "vectors.npz",
        target_vector_count=2,
        embedder=HashingTextEmbedder(dimensions=16),
        fetch_text=lambda _url: _many_records_xml(5),
        embedding_batch_size=2,
    )

    conn = connect_db(tmp_path / "papers.db")
    index = ExactVectorIndex.load(tmp_path / "vectors.npz")

    assert summary.records_seen == 2
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 2
    assert index.vector_ids.tolist() == [1, 2]


def test_build_index_from_oai_commits_embedding_chunks_in_batches(tmp_path) -> None:
    db_path = tmp_path / "papers.db"

    build_index_from_oai(
        endpoint="https://example.test/oai",
        db_path=db_path,
        index_path=tmp_path / "vectors.npz",
        embedder=HashingTextEmbedder(dimensions=16),
        fetch_text=lambda _url: _many_records_xml(3),
        embedding_batch_size=2,
    )

    conn = connect_db(db_path)

    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 3

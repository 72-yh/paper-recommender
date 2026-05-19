import numpy as np

from paper_recommender.index_builder import build_index_from_oai
from paper_recommender.embedding import HashingTextEmbedder
from paper_recommender.models import Paper
from paper_recommender.recommender import recommend
from paper_recommender.storage import connect_db, get_paper, init_db, upsert_paper
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

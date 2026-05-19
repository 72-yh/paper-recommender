import numpy as np

from paper_recommender.storage import connect_db, get_paper
from paper_recommender.vector_store import ExactVectorIndex
from scripts.seed_sample_data import main


def test_seed_sample_data_writes_expected_rows_and_vectors(tmp_path) -> None:
    main(tmp_path)

    conn = connect_db(tmp_path / "paper_recommender.db")
    papers = {
        arxiv_id: get_paper(conn, arxiv_id)
        for arxiv_id in ("1706.03762", "1111.11111", "2222.22222", "3333.33333")
    }
    index = ExactVectorIndex.load(tmp_path / "vectors.npz")

    assert (tmp_path / "paper_recommender.db").exists()
    assert (tmp_path / "vectors.npz").exists()
    assert len(conn.execute("SELECT arxiv_id FROM papers").fetchall()) == 4
    assert papers["1706.03762"] is not None
    assert papers["1706.03762"].vector_id == 1
    assert papers["1706.03762"].categories == ("cs.CL", "cs.LG")
    assert papers["1111.11111"].primary_category == "cs.CL"
    assert papers["2222.22222"].primary_category == "cs.LG"
    assert papers["3333.33333"].primary_category == "stat.ML"
    assert np.allclose(index.get(1), np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(index.get(2), np.array([0.9938837, 0.11043153, 0.0], dtype=np.float32))
    assert np.allclose(index.get(3), np.array([0.24253564, 0.97014254, 0.0], dtype=np.float32))
    assert np.allclose(index.get(4), np.array([0.0, 0.0, 1.0], dtype=np.float32))

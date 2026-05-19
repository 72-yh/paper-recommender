from __future__ import annotations

from pathlib import Path

import numpy as np

from paper_recommender.models import Paper
from paper_recommender.storage import connect_db, init_db, upsert_paper
from paper_recommender.vector_store import ExactVectorIndex


def main(data_dir: Path = Path("data")) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "paper_recommender.db"
    vector_path = data_dir / "vectors.npz"

    conn = connect_db(db_path)
    try:
        init_db(conn)
        for paper in _sample_papers():
            upsert_paper(conn, paper)
    finally:
        conn.close()

    index = ExactVectorIndex.from_items(
        {
            1: np.array([1.0, 0.0, 0.0], dtype=np.float32),
            2: np.array([0.9, 0.1, 0.0], dtype=np.float32),
            3: np.array([0.2, 0.8, 0.0], dtype=np.float32),
            4: np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    )
    index.save(vector_path)

    print(f"Wrote {db_path} and {vector_path}")


def _sample_papers() -> list[Paper]:
    return [
        Paper(
            arxiv_id="1706.03762",
            vector_id=1,
            active=True,
            oai_datestamp="2024-01-01",
            published_date="2017-06-12",
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL", "cs.LG"),
            content_hash="sample-1706.03762",
        ),
        Paper(
            arxiv_id="1111.11111",
            vector_id=2,
            active=True,
            oai_datestamp="2024-01-01",
            published_date="2011-11-11",
            updated_date=None,
            primary_category="cs.CL",
            categories=("cs.CL",),
            content_hash="sample-1111.11111",
        ),
        Paper(
            arxiv_id="2222.22222",
            vector_id=3,
            active=True,
            oai_datestamp="2024-01-01",
            published_date="2022-02-22",
            updated_date=None,
            primary_category="cs.LG",
            categories=("cs.LG",),
            content_hash="sample-2222.22222",
        ),
        Paper(
            arxiv_id="3333.33333",
            vector_id=4,
            active=True,
            oai_datestamp="2024-01-01",
            published_date="2033-03-03",
            updated_date=None,
            primary_category="stat.ML",
            categories=("stat.ML",),
            content_hash="sample-3333.33333",
        ),
    ]


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from paper_recommender.arxiv_id import InvalidArxivUrl, parse_arxiv_id
from paper_recommender.compressed_vector_store import Int8VectorIndex, MmapInt8VectorIndex
from paper_recommender.recommender import RecommendationError, recommend
from paper_recommender.storage import connect_db, get_pipeline_state
from paper_recommender.vector_store import ExactVectorIndex


class RecommendRequest(BaseModel):
    url: str
    category: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)


class IndexStatus(BaseModel):
    active_papers: int
    indexed_papers: int
    last_oai_datestamp: str | None
    index_kind: str
    index_bytes: int


def create_app(
    db_path: str | Path | None = None,
    index_path: str | Path | None = None,
    index_kind: str | None = None,
) -> FastAPI:
    db_path = db_path or os.environ.get("PAPER_RECOMMENDER_DB_PATH", "data/paper_recommender.db")
    index_path = index_path or os.environ.get("PAPER_RECOMMENDER_INDEX_PATH", "data/vectors.npz")
    index_kind = index_kind or os.environ.get("PAPER_RECOMMENDER_INDEX_KIND", "exact")
    app = FastAPI(title="Paper Recommender")
    cached_index = None
    index_lock = threading.Lock()

    def get_index():
        nonlocal cached_index
        if cached_index is None:
            with index_lock:
                if cached_index is None:
                    cached_index = _load_index(index_path, index_kind)
        return cached_index

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    def index_status() -> IndexStatus:
        conn = connect_db(db_path)
        try:
            active_papers = _count_papers(conn, "active = 1")
            indexed_papers = _count_papers(conn, "active = 1 AND vector_id IS NOT NULL")
            last_oai_datestamp = get_pipeline_state(conn, "last_successful_oai_datestamp")
            if last_oai_datestamp is None:
                last_oai_datestamp = _max_oai_datestamp(conn)
        finally:
            conn.close()

        return IndexStatus(
            active_papers=active_papers,
            indexed_papers=indexed_papers,
            last_oai_datestamp=last_oai_datestamp,
            index_kind=index_kind,
            index_bytes=_file_size(index_path),
        )

    @app.post("/api/recommend")
    def recommend_papers(request: RecommendRequest) -> dict[str, object]:
        try:
            arxiv_id = parse_arxiv_id(request.url)
            index = get_index()
            conn = connect_db(db_path)
            try:
                results = recommend(
                    conn,
                    index,
                    arxiv_id,
                    top_k=request.top_k,
                    category=request.category,
                    date_from=request.date_from,
                    date_to=request.date_to,
                )
            finally:
                conn.close()
        except InvalidArxivUrl as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RecommendationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

        return {"query_arxiv_id": arxiv_id, "results": results}

    static_dir = Path(__file__).with_name("static")
    if static_dir.exists():
        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


def _load_index(index_path: str | Path, index_kind: str):
    if index_kind == "exact":
        return ExactVectorIndex.load(index_path)
    if index_kind == "int8":
        return Int8VectorIndex.load(index_path)
    if index_kind == "int8_mmap":
        return MmapInt8VectorIndex.load(index_path)
    raise RecommendationError(500, f"Unsupported vector index kind: {index_kind}")


def _count_papers(conn, where_clause: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS value FROM papers WHERE {where_clause}").fetchone()
    return int(row["value"])


def _max_oai_datestamp(conn) -> str | None:
    row = conn.execute("SELECT MAX(oai_datestamp) AS value FROM papers").fetchone()
    return None if row is None else row["value"]


def _file_size(path: str | Path) -> int:
    try:
        return Path(path).stat().st_size
    except FileNotFoundError:
        return 0


app = create_app()

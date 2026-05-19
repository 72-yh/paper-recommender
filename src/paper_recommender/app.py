from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from paper_recommender.arxiv_id import InvalidArxivUrl, parse_arxiv_id
from paper_recommender.recommender import RecommendationError, recommend
from paper_recommender.storage import connect_db
from paper_recommender.vector_store import ExactVectorIndex


class RecommendRequest(BaseModel):
    url: str
    category: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)


def create_app(
    db_path: str | Path = "data/paper_recommender.db",
    index_path: str | Path = "data/vectors.npz",
) -> FastAPI:
    app = FastAPI(title="Paper Recommender")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/recommend")
    def recommend_papers(request: RecommendRequest) -> dict[str, object]:
        try:
            arxiv_id = parse_arxiv_id(request.url)
            index = ExactVectorIndex.load(index_path)
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


app = create_app()

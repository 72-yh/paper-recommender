from __future__ import annotations

import sqlite3
from typing import Protocol

import numpy as np

from paper_recommender.models import (
    DELETED_RECORD_MESSAGE,
    Recommendation,
    UNKNOWN_ID_MESSAGE,
    VECTOR_MISSING_MESSAGE,
)
from paper_recommender.storage import get_paper, get_paper_by_vector_id
from paper_recommender.vector_store import VectorSearchResult


class RecommendationError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message)


class SearchableIndex(Protocol):
    def get(self, vector_id: int) -> np.ndarray | None: ...

    def search(self, query: np.ndarray, top_k: int) -> list[VectorSearchResult]: ...


def recommend(
    conn: sqlite3.Connection,
    index: SearchableIndex,
    arxiv_id: str,
    top_k: int,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Recommendation]:
    if top_k <= 0:
        return []

    query_paper = get_paper(conn, arxiv_id)
    if query_paper is None:
        raise RecommendationError(404, UNKNOWN_ID_MESSAGE)
    if not query_paper.active:
        raise RecommendationError(404, DELETED_RECORD_MESSAGE)
    if query_paper.vector_id is None:
        raise RecommendationError(404, VECTOR_MISSING_MESSAGE)

    query_vector = index.get(query_paper.vector_id)
    if query_vector is None:
        raise RecommendationError(404, VECTOR_MISSING_MESSAGE)

    recommendations: list[Recommendation] = []
    for vector_result in index.search(query_vector, top_k=max(top_k * 20, 100)):
        candidate = get_paper_by_vector_id(conn, vector_result.vector_id)
        if candidate is None:
            continue
        if candidate.arxiv_id == arxiv_id:
            continue
        if not candidate.active:
            continue
        if category is not None and category not in candidate.categories:
            continue
        if (date_from is not None or date_to is not None) and candidate.published_date is None:
            continue
        if date_from is not None and candidate.published_date is not None:
            if candidate.published_date < date_from:
                continue
        if date_to is not None and candidate.published_date is not None:
            if candidate.published_date > date_to:
                continue

        recommendations.append(
            Recommendation(
                arxiv_id=candidate.arxiv_id,
                url=f"https://arxiv.org/abs/{candidate.arxiv_id}",
                primary_category=candidate.primary_category,
                categories=candidate.categories,
                published_date=candidate.published_date,
                updated_date=candidate.updated_date,
                similarity_score=vector_result.score,
            )
        )
        if len(recommendations) == top_k:
            break

    return recommendations

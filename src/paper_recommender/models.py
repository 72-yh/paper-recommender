from __future__ import annotations

from dataclasses import dataclass

UNKNOWN_ID_MESSAGE = (
    "This arXiv ID is not available in the recommendation index yet. "
    "It may not have been backfilled or may have been removed."
)
VECTOR_MISSING_MESSAGE = (
    "The recommendation vector for this paper is not ready yet. "
    "Please try again after the next index update."
)
DELETED_RECORD_MESSAGE = (
    "This paper is marked as deleted in arXiv OAI-PMH and is excluded from recommendations."
)
NOT_ENOUGH_RESULTS_MESSAGE = (
    "Not enough similar papers match your filters. Try relaxing the category or date range."
)


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    vector_id: int | None
    active: bool
    oai_datestamp: str
    published_date: str | None
    updated_date: str | None
    primary_category: str
    categories: tuple[str, ...]
    content_hash: str


@dataclass(frozen=True)
class Recommendation:
    arxiv_id: str
    url: str
    primary_category: str
    categories: tuple[str, ...]
    published_date: str | None
    updated_date: str | None
    similarity_score: float

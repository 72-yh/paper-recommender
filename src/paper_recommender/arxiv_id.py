from __future__ import annotations

import re
from urllib.parse import urlparse

INVALID_ARXIV_URL_MESSAGE = (
    "Please enter a valid arXiv URL, e.g. https://arxiv.org/abs/1706.03762"
)

_MODERN_ID_RE = re.compile(r"^(?P<id>\d{4}\.\d{4,5})(?:v\d+)?$")
_OLD_ID_RE = re.compile(r"^(?P<id>[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")


class InvalidArxivUrl(ValueError):
    """Raised when a URL does not point to a supported arXiv abs or pdf path."""


def parse_arxiv_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise InvalidArxivUrl(INVALID_ARXIV_URL_MESSAGE)
    if parsed.netloc.lower() not in {"arxiv.org", "www.arxiv.org"}:
        raise InvalidArxivUrl(INVALID_ARXIV_URL_MESSAGE)

    path = parsed.path.strip("/")
    if path.startswith("abs/"):
        raw_id = path.removeprefix("abs/")
    elif path.startswith("pdf/"):
        raw_id = path.removeprefix("pdf/")
        raw_id = raw_id.removesuffix(".pdf")
    else:
        raise InvalidArxivUrl(INVALID_ARXIV_URL_MESSAGE)

    for pattern in (_MODERN_ID_RE, _OLD_ID_RE):
        match = pattern.match(raw_id)
        if match:
            return match.group("id")

    raise InvalidArxivUrl(INVALID_ARXIV_URL_MESSAGE)

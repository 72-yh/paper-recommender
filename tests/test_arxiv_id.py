import pytest

from paper_recommender.arxiv_id import InvalidArxivUrl, parse_arxiv_id


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
        ("https://arxiv.org/abs/1706.03762v7", "1706.03762"),
        ("https://www.arxiv.org/abs/cs/9901001", "cs/9901001"),
        ("https://arxiv.org/pdf/hep-th/9901001.pdf", "hep-th/9901001"),
    ],
)
def test_parse_supported_arxiv_urls(url: str, expected: str) -> None:
    assert parse_arxiv_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",
        "1706.03762",
        "https://example.com/abs/1706.03762",
        "https://arxiv.org/list/cs.AI/recent",
        "https://arxiv.org/abs/not-an-id",
    ],
)
def test_reject_invalid_urls(url: str) -> None:
    with pytest.raises(InvalidArxivUrl) as exc_info:
        parse_arxiv_id(url)

    assert "Please enter a valid arXiv URL" in str(exc_info.value)

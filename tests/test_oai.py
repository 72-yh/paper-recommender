from pathlib import Path

from paper_recommender.oai import parse_oai_records


def test_parse_normal_and_deleted_records() -> None:
    xml = Path("tests/fixtures/oai_records.xml").read_text(encoding="utf-8")

    batch = parse_oai_records(xml)

    assert batch.resumption_token == "abc123"
    assert len(batch.records) == 2
    assert batch.records[0].arxiv_id == "1706.03762"
    assert batch.records[0].deleted is False
    assert batch.records[0].title == "Attention Is All You Need"
    assert batch.records[0].categories == ("cs.CL", "cs.LG")
    assert batch.records[1].arxiv_id == "9999.00001"
    assert batch.records[1].deleted is True

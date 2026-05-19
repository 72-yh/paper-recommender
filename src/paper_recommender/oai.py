from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET


OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "ax": "http://arxiv.org/OAI/arXiv/",
}


@dataclass(frozen=True)
class OaiRecord:
    arxiv_id: str
    oai_datestamp: str
    deleted: bool
    title: str | None
    abstract: str | None
    categories: tuple[str, ...]
    published_date: str | None
    updated_date: str | None


@dataclass(frozen=True)
class OaiBatch:
    records: tuple[OaiRecord, ...]
    resumption_token: str | None


def _text(parent: ET.Element, path: str) -> str | None:
    node = parent.find(path, OAI_NS)
    if node is None or node.text is None:
        return None
    return " ".join(node.text.split())


def _arxiv_id_from_header(identifier: str) -> str:
    return identifier.removeprefix("oai:arXiv.org:")


def parse_oai_records(xml: str) -> OaiBatch:
    root = ET.fromstring(xml)
    records: list[OaiRecord] = []
    for record in root.findall(".//oai:record", OAI_NS):
        header = record.find("oai:header", OAI_NS)
        if header is None:
            continue
        identifier = _text(header, "oai:identifier")
        datestamp = _text(header, "oai:datestamp")
        if identifier is None or datestamp is None:
            continue
        arxiv_id = _arxiv_id_from_header(identifier)
        deleted = header.attrib.get("status") == "deleted"
        if deleted:
            records.append(
                OaiRecord(
                    arxiv_id=arxiv_id,
                    oai_datestamp=datestamp,
                    deleted=True,
                    title=None,
                    abstract=None,
                    categories=(),
                    published_date=None,
                    updated_date=None,
                )
            )
            continue

        metadata = record.find("oai:metadata/ax:arXiv", OAI_NS)
        if metadata is None:
            continue
        categories_text = _text(metadata, "ax:categories") or ""
        records.append(
            OaiRecord(
                arxiv_id=_text(metadata, "ax:id") or arxiv_id,
                oai_datestamp=datestamp,
                deleted=False,
                title=_text(metadata, "ax:title") or "",
                abstract=_text(metadata, "ax:abstract") or "",
                categories=tuple(part for part in categories_text.split(" ") if part),
                published_date=_text(metadata, "ax:created"),
                updated_date=_text(metadata, "ax:updated"),
            )
        )

    token = _text(root, ".//oai:resumptionToken")
    return OaiBatch(records=tuple(records), resumption_token=token)

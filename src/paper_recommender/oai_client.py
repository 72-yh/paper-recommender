from __future__ import annotations

from collections.abc import Callable, Iterator
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from paper_recommender.oai import OaiBatch, parse_oai_records


OAI_ENDPOINT = "https://export.arxiv.org/oai2"
USER_AGENT = "paper-recommender/0.1 local-index-proof"


def build_list_records_url(
    endpoint: str = OAI_ENDPOINT,
    *,
    from_date: str | None = None,
    until_date: str | None = None,
    resumption_token: str | None = None,
) -> str:
    if resumption_token is not None:
        query = {"verb": "ListRecords", "resumptionToken": resumption_token}
    else:
        query = {"verb": "ListRecords", "metadataPrefix": "arXiv"}
        if from_date is not None:
            query["from"] = from_date
        if until_date is not None:
            query["until"] = until_date

    return f"{endpoint}?{urlencode(query)}"


def fetch_oai_batches(
    endpoint: str = OAI_ENDPOINT,
    *,
    from_date: str | None = None,
    until_date: str | None = None,
    batch_limit: int | None = None,
    fetch_text: Callable[[str], str] | None = None,
    request_delay_seconds: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[OaiBatch]:
    fetch = fetch_text or _fetch_text
    resumption_token: str | None = None
    batches_seen = 0

    while True:
        if batch_limit is not None and batches_seen >= batch_limit:
            return

        url = build_list_records_url(
            endpoint,
            from_date=from_date,
            until_date=until_date,
            resumption_token=resumption_token,
        )
        batch = parse_oai_records(fetch(url))
        yield batch

        batches_seen += 1
        if not batch.resumption_token:
            return
        resumption_token = batch.resumption_token
        if request_delay_seconds > 0:
            sleep(request_delay_seconds)


def _fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")

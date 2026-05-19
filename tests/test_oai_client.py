from paper_recommender.oai_client import build_list_records_url, fetch_oai_batches


def test_build_list_records_url_uses_datestamp_window() -> None:
    url = build_list_records_url(
        "https://example.test/oai",
        from_date="2024-01-01",
        until_date="2024-01-02",
    )

    assert url == (
        "https://example.test/oai?"
        "verb=ListRecords&metadataPrefix=arXiv&from=2024-01-01&until=2024-01-02"
    )


def test_build_list_records_url_uses_only_resumption_token_when_present() -> None:
    url = build_list_records_url(
        "https://example.test/oai",
        from_date="2024-01-01",
        until_date="2024-01-02",
        resumption_token="abc 123",
    )

    assert url == "https://example.test/oai?verb=ListRecords&resumptionToken=abc+123"


def test_fetch_oai_batches_follows_resumption_token() -> None:
    responses = {
        "https://example.test/oai?verb=ListRecords&metadataPrefix=arXiv&from=2024-01-01": """
            <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
              <ListRecords>
                <record>
                  <header>
                    <identifier>oai:arXiv.org:1706.03762</identifier>
                    <datestamp>2024-01-01</datestamp>
                  </header>
                  <metadata>
                    <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
                      <id>1706.03762</id>
                      <created>2017-06-12</created>
                      <title>Attention Is All You Need</title>
                      <abstract>Transformer networks.</abstract>
                      <categories>cs.CL</categories>
                    </arXiv>
                  </metadata>
                </record>
                <resumptionToken>next-token</resumptionToken>
              </ListRecords>
            </OAI-PMH>
        """,
        "https://example.test/oai?verb=ListRecords&resumptionToken=next-token": """
            <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
              <ListRecords>
                <record>
                  <header status="deleted">
                    <identifier>oai:arXiv.org:9999.00001</identifier>
                    <datestamp>2024-01-02</datestamp>
                  </header>
                </record>
              </ListRecords>
            </OAI-PMH>
        """,
    }
    requested_urls: list[str] = []

    def fetch_text(url: str) -> str:
        requested_urls.append(url)
        return responses[url]

    batches = list(
        fetch_oai_batches(
            "https://example.test/oai",
            from_date="2024-01-01",
            fetch_text=fetch_text,
        )
    )

    assert requested_urls == list(responses)
    assert [batch.records[0].arxiv_id for batch in batches] == ["1706.03762", "9999.00001"]


def test_fetch_oai_batches_honors_batch_limit() -> None:
    xml = """
        <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
          <ListRecords><resumptionToken>next-token</resumptionToken></ListRecords>
        </OAI-PMH>
    """

    batches = list(
        fetch_oai_batches(
            "https://example.test/oai",
            from_date="2024-01-01",
            batch_limit=1,
            fetch_text=lambda _url: xml,
        )
    )

    assert len(batches) == 1


def test_fetch_oai_batches_delays_between_followup_requests() -> None:
    responses = {
        "https://example.test/oai?verb=ListRecords&metadataPrefix=arXiv&from=2024-01-01": """
            <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
              <ListRecords><resumptionToken>next-token</resumptionToken></ListRecords>
            </OAI-PMH>
        """,
        "https://example.test/oai?verb=ListRecords&resumptionToken=next-token": """
            <OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
              <ListRecords></ListRecords>
            </OAI-PMH>
        """,
    }
    delays: list[float] = []

    list(
        fetch_oai_batches(
            "https://example.test/oai",
            from_date="2024-01-01",
            fetch_text=lambda url: responses[url],
            request_delay_seconds=3.0,
            sleep=delays.append,
        )
    )

    assert delays == [3.0]

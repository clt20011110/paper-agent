from __future__ import annotations

import gzip

from paper_agent.stage1_hydration import (
    _aaai_oai_page,
    _legacy_virtual_poster,
    _virtual_openreview_id,
)


def test_aaai_oai_page_extracts_abstract_doi_pdf_and_cursor() -> None:
    body = b'''<?xml version="1.0"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
 xmlns:dc="http://purl.org/dc/elements/1.1/">
 <ListRecords><record><header><identifier>oai:ojs.aaai.org:article/31001</identifier></header>
 <metadata><dc:dc><dc:description>Official abstract.</dc:description>
 <dc:identifier>10.1609/aaai.v38i1.31001</dc:identifier>
 <dc:relation>https://ojs.aaai.org/index.php/AAAI/article/view/31001/33001</dc:relation>
 </dc:dc></metadata></record><resumptionToken>next-page</resumptionToken></ListRecords>
</OAI-PMH>'''

    records, cursor = _aaai_oai_page(body)

    assert cursor == "next-page"
    assert records["31001"] == {
        "abstract": "Official abstract.",
        "doi": "10.1609/aaai.v38i1.31001",
        "pdf_url": "https://ojs.aaai.org/index.php/AAAI/article/view/31001/33001",
        "_source": "aaai_oai",
    }


def test_iclr_virtual_and_legacy_pages_extract_openreview_id_and_abstract() -> None:
    record = {
        "paper_url": "https://openreview.net/forum?id=Forum123",
        "eventmedia": [],
    }
    page = b'''<a href="https://openreview.net/forum?id=Forum123">PDF</a>
<div class="abstract-text-inner"><p>A &amp; B official abstract.</p></div>'''

    assert _virtual_openreview_id(record) == "Forum123"
    assert _legacy_virtual_poster(page) == {
        "forum_id": "Forum123",
        "abstract": "A & B official abstract.",
    }


def test_gzip_fixture_documents_aaai_missing_content_encoding_case() -> None:
    payload = b"<html><title>AAAI article</title></html>"
    assert gzip.decompress(gzip.compress(payload)) == payload

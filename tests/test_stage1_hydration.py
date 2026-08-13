from __future__ import annotations

import gzip
from io import BytesIO
import tarfile

from paper_agent.stage1_hydration import (
    _aaai_oai_page,
    _legacy_virtual_poster,
    _ijcai_detail,
    _pmlr_frontmatter_snapshot,
    _neurips_export_records,
    _neurips_detail_abstract,
    _neurips_crossref_page,
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


def test_ijcai_detail_extracts_official_abstract_doi_and_pdf() -> None:
    body = b'''<meta name="citation_pdf_url" content="https://www.ijcai.org/proceedings/2024/0001.pdf" />
<a href="https://doi.org/10.24963/ijcai.2024/1" class="doi">DOI</a>
<div class="col-md-12">A certified policy verification abstract.</div>'''

    assert _ijcai_detail(body) == {
        "abstract": "A certified policy verification abstract.",
        "doi": "10.24963/ijcai.2024/1",
        "pdf_url": "https://www.ijcai.org/proceedings/2024/0001.pdf",
        "_source": "ijcai_official:paper_detail",
    }


def test_ijcai_2016_detail_extracts_legacy_official_abstract_and_pdf() -> None:
    body = b'''<div class="content"><html><body>
<p>Auditable Artificial Intelligence / 2<br /><i>Ada Lovelace</i></p>
<p>A complete &amp; official legacy abstract.</p>
<p><a href="/Proceedings/16/Papers/008.pdf">PDF</a></p>
</body></html></div>'''

    assert _ijcai_detail(body) == {
        "abstract": "A complete & official legacy abstract.",
        "doi": None,
        "pdf_url": "https://www.ijcai.org/Proceedings/16/Papers/008.pdf",
        "_source": "ijcai_official:paper_detail",
    }


def test_pmlr_snapshot_extracts_frontmatter_without_pdf_fetch() -> None:
    markdown = b'''---
id: audit24a
title: Auditable Metadata
abstract: A complete official abstract.
pdf: https://raw.githubusercontent.com/mlresearch/v235/main/assets/audit24a/audit24a.pdf
---
'''
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("v235-gh-pages/_posts/2024-07-08-audit24a.md")
        info.size = len(markdown)
        archive.addfile(info, BytesIO(markdown))

    assert _pmlr_frontmatter_snapshot(buffer.getvalue(), "v235") == {
        "v235/audit24a": {
            "id": "audit24a",
            "title": "Auditable Metadata",
            "abstract": "A complete official abstract.",
            "pdf": "https://raw.githubusercontent.com/mlresearch/v235/main/assets/audit24a/audit24a.pdf",
        }
    }


def test_neurips_export_keeps_duplicate_titles_for_ambiguous_join_detection() -> None:
    records = _neurips_export_records(b'''[
      {"type":"Poster","name":"Same Paper","abstract":"One","virtualsite_url":"/1"},
      {"type":"Poster","name":"Same Paper","abstract":"Two","virtualsite_url":"/2"},
      {"type":"Tutorial","name":"Ignored","abstract":"Not a paper"}
    ]''')

    assert [record["abstract"] for record in records["samepaper"]] == ["One", "Two"]


def test_neurips_detail_extracts_official_abstract_fallback() -> None:
    assert _neurips_detail_abstract(
        b'<p class="paper-abstract"><p>A &amp; B <em>official</em> abstract.</p>'
    ) == "A & B official abstract."


def test_neurips_crossref_page_filters_exact_proceedings_container() -> None:
    body = b'''{"message":{"items":[
      {"DOI":"10.52202/079017-0001","title":["Auditable Paper"],
       "container-title":["Advances in Neural Information Processing Systems 37"]},
      {"DOI":"10.52202/other","title":["Other"],"container-title":["Other"]}
    ],"next-cursor":"next"}}'''

    assert _neurips_crossref_page(
        body, "Advances in Neural Information Processing Systems 37"
    ) == ({"auditablepaper": "10.52202/079017-0001"}, None)

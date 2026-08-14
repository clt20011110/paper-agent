from __future__ import annotations

import gzip
from http.client import IncompleteRead
from io import BytesIO
import tarfile
from types import SimpleNamespace

from paper_agent.stage1_hydration import (
    OfficialStage1FieldHydrator,
    _aaai_oai_page,
    _legacy_virtual_poster,
    _ijcai_detail,
    _pmlr_frontmatter_snapshot,
    _pmlr_detail,
    _neurips_export_records,
    _neurips_detail_abstract,
    _neurips_crossref_page,
    _jmlr_rss_records,
    _jmlr_detail,
    _acl_crossref_page,
    _cvf_crossref_exact_doi,
    _cvf_dblp_dois,
    _cvf_detail,
    _cvf_virtual_records,
    _eda_semantic_scholar_batch,
    _openalex_abstract,
    _eda_publisher_pdf_url,
    _europe_pmc_doi_records,
    _journal_publisher_pdf_url,
    _journal_resource_pdf_url,
    _nature_article_abstract,
    _nature_article_document_type,
    _virtual_openreview_id,
)
from paper_agent.domain import SourceEntry


def test_cvf_bulk_and_detail_sources_extract_required_fields() -> None:
    virtual = _cvf_virtual_records(b'''{"results":[
      {"eventtype":"Poster","name":"Auditable Vision","abstract":"Bulk abstract."},
      {"eventtype":"Workshop","name":"Excluded","abstract":"No."}
    ]}''')
    assert virtual["auditablevision"][0]["abstract"] == "Bulk abstract."

    detail = _cvf_detail(b'''<meta name="citation_pdf_url" content="https://cvf/paper.pdf">
      <div id="abstract">Official &amp; complete <em>abstract</em>.</div>''')
    assert detail == {
        "abstract": "Official & complete abstract.",
        "pdf_url": "https://cvf/paper.pdf",
        "_source": "cvf_open_access:paper_detail.abstract",
    }


def test_cvf_dblp_and_crossref_join_only_registered_conference_doi() -> None:
    dblp = _cvf_dblp_dois(b'''<li class="entry inproceedings">
      <a href="https://doi.org/10.1109/CVPR1.2024.00001">DOI</a>
      <span class="title">Auditable &amp; Vision.</span></li>''')
    assert dblp["auditablevision"][0]["doi"] == "10.1109/cvpr1.2024.00001"

    crossref = b'''{"message":{"items":[
      {"DOI":"10.1109/CVPR1.2024.00001","title":["Auditable Vision"],
       "container-title":["2024 IEEE/CVF Conference (CVPR)"]},
      {"DOI":"10.1109/OTHER","title":["Auditable Vision"],
       "container-title":["Other conference"]}
    ]}}'''
    assert _cvf_crossref_exact_doi(crossref, "CVPR", "Auditable Vision") == (
        "10.1109/cvpr1.2024.00001"
    )


def test_eda_semantic_scholar_batch_joins_abstract_and_oa_pdf_by_doi() -> None:
    records = _eda_semantic_scholar_batch(b'''[
      {"paperId":"p1","externalIds":{"DOI":"10.1145/123.456"},
       "abstract":"Officially indexed abstract.",
       "openAccessPdf":{"url":"https://arxiv.org/pdf/1234.5678"}},
      null
    ]''')
    assert records == {
        "10.1145/123.456": {
            "abstract": "Officially indexed abstract.",
            "pdf_url": "https://arxiv.org/pdf/1234.5678",
            "abstract_source": "semantic_scholar:doi_batch.abstract",
        }
    }
    assert _eda_publisher_pdf_url("10.1145/123.456") == (
        "https://dl.acm.org/doi/pdf/10.1145/123.456"
    )


def test_eda_timeout_keeps_membership_and_records_recoverable_warnings() -> None:
    class Transport:
        def fetch_json_batch(self, provider, url, **kwargs):
            assert provider == "semantic_scholar"
            raise TimeoutError("scholarly graph timed out")

        def fetch_metadata(self, provider, url, **kwargs):
            assert provider in {"openalex", "arxiv"}
            raise TimeoutError(f"{provider} timed out")

    entry = SourceEntry(
        "dblp_toc",
        "conf/date/Timeout24",
        "A Timeout-Tolerant EDA Paper",
        doi="10.1145/123.456",
        pdf_url="https://example.org/paper.pdf",
    )
    result = OfficialStage1FieldHydrator(Transport())._eda((entry,))

    assert [item.external_id for item in result.entries] == [entry.external_id]
    assert result.entries[0].abstract is None
    assert any("Semantic Scholar fallback unavailable" in warning for warning in result.warnings)
    assert any("OpenAlex fallback unavailable" in warning for warning in result.warnings)
    assert any("arXiv fallback unavailable" in warning for warning in result.warnings)


def test_crossref_europe_pmc_timeout_continues_to_semantic_scholar() -> None:
    class Transport:
        def fetch_metadata(self, provider, url, **kwargs):
            assert provider == "europe_pmc"
            raise TimeoutError("Europe PMC timed out")

        def fetch_json_batch(self, provider, url, **kwargs):
            assert provider == "semantic_scholar"
            return SimpleNamespace(
                body=b'''[{"paperId":"p1","externalIds":{"DOI":"10.1021/example"},
                    "abstract":"Fallback abstract.",
                    "openAccessPdf":{"url":"https://example.org/fallback.pdf"}}]'''
            )

    entry = SourceEntry(
        "crossref_serial",
        "10.1021/example",
        "A Fallback Journal Paper",
        doi="10.1021/example",
    )
    result = OfficialStage1FieldHydrator(Transport())._crossref_journal(
        SimpleNamespace(parameters={}), (entry,)
    )

    assert result.entries[0].abstract == "Fallback abstract."
    assert result.entries[0].pdf_url == "https://example.org/fallback.pdf"
    assert any("Europe PMC batch 0 unavailable" in warning for warning in result.warnings)


def test_openalex_abstract_reconstructs_inverted_index() -> None:
    assert _openalex_abstract({"official": [2], "Exact": [0], "abstract": [1]}) == (
        "Exact abstract official"
    )


def test_openalex_title_fallback_requires_exact_doi_and_title() -> None:
    class Transport:
        def fetch_metadata(self, provider, url, **kwargs):
            assert provider == "openalex"
            return type(
                "Response",
                (),
                {
                    "body": b'''{"results":[
                      {"doi":"https://doi.org/10.1145/123.456",
                       "title":"Auditable Molecular Generation",
                       "abstract_inverted_index":{"official":[2],"Exact":[0],"abstract":[1]}},
                      {"doi":"https://doi.org/10.1145/other",
                       "title":"Auditable Molecular Generation",
                       "abstract_inverted_index":{"wrong":[0]}}
                    ]}'''
                },
            )()

    entry = SourceEntry("dblp_toc", "conf/dac/Auditable24", "Auditable Molecular Generation", doi="10.1145/123.456")
    entry_result, record, body_hash, error = OfficialStage1FieldHydrator(Transport())._openalex_title_record_safe(entry)

    assert entry_result is entry
    assert record == {
        "abstract": "Exact abstract official",
        "abstract_source": "openalex:works.search.abstract_inverted_index",
    }
    assert body_hash
    assert error is None


def test_europe_pmc_doi_batch_extracts_abstract_and_open_pdf() -> None:
    records = _europe_pmc_doi_records(b'''{"resultList":{"result":[{
      "doi":"10.1021/example","abstractText":"Indexed abstract.",
      "fullTextUrlList":{"fullTextUrl":[
        {"availabilityCode":"S","documentStyle":"doi","url":"https://doi.org/x"},
        {"availabilityCode":"OA","documentStyle":"pdf","url":"https://pmc/x.pdf"}
      ]}}]}}''')
    assert records["10.1021/example"]["abstract"] == "Indexed abstract."
    assert records["10.1021/example"]["pdf_url"] == "https://pmc/x.pdf"


def test_europe_pmc_doi_batch_preserves_publisher_document_types() -> None:
    records = _europe_pmc_doi_records(b'''{"resultList":{"result":[{
      "doi":"10.1021/news","pubType":"News",
      "pubTypeList":{"pubType":["News","Journal Article"]}
    }]}}''')
    assert records["10.1021/news"]["document_type"] == "News"
    assert records["10.1021/news"]["document_types"] == (
        "News", "Journal Article"
    )


def test_journal_arxiv_fallback_requires_exact_normalized_title() -> None:
    class Transport:
        def fetch_metadata(self, provider, url, **kwargs):
            assert provider == "arxiv"
            return type("Response", (), {"body": b'''<feed xmlns="http://www.w3.org/2005/Atom">
              <entry><id>https://arxiv.org/abs/2401.00001v2</id>
              <title>Auditable molecular generation</title>
              <summary>Public preprint abstract.</summary></entry>
              <entry><id>https://arxiv.org/abs/2401.00002</id>
              <title>A similar but different paper</title>
              <summary>Must not be joined.</summary></entry>
            </feed>'''})()

    entries = (
        SourceEntry("crossref_serial", "10.1/a", "Auditable Molecular Generation"),
        SourceEntry("crossref_serial", "10.1/b", "No Exact Match"),
    )
    records, hashes = OfficialStage1FieldHydrator(Transport())._arxiv_title_fallback(
        entries, "journal"
    )

    assert set(records) == {"10.1/a"}
    assert records["10.1/a"] == {
        "abstract": "Public preprint abstract.",
        "pdf_url": "https://arxiv.org/pdf/2401.00001",
        "abstract_source": "arxiv_atom:exact_title_summary",
        "pdf_source": "arxiv_atom:public_pdf",
    }
    assert len(hashes) == 1


def test_science_registered_doi_has_canonical_publisher_pdf_route() -> None:
    assert _journal_publisher_pdf_url("10.1126/science.example") == (
        "https://www.science.org/doi/pdf/10.1126/science.example"
    )
    assert _journal_publisher_pdf_url("10.1002/anie.example") is None
    assert _journal_publisher_pdf_url("10.1039/d4sc01391c") == (
        "https://pubs.rsc.org/en/content/articlepdf/2024/sc/d4sc01391c"
    )
    assert _journal_publisher_pdf_url("10.1039/D3SC90029K") == (
        "https://pubs.rsc.org/en/content/articlepdf/2023/sc/d3sc90029k"
    )
    assert _journal_resource_pdf_url(
        {"resource_url": "https://linkinghub.elsevier.com/retrieve/pii/S0092867424007001"},
        "10.1016/j.cell.example",
    ) == "https://www.cell.com/action/showPdf?pii=S0092867424007001"
    assert _journal_resource_pdf_url(
        {"resource_url": "https://linkinghub.elsevier.com/retrieve/pii/S1234567890"},
        "10.1016/j.other.example",
    ) is None
    assert _journal_resource_pdf_url(
        {"resource_url": "https://ieeexplore.ieee.org/document/10439997/"},
        "10.1109/tcad.example",
    ) == "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10439997"


def test_nature_article_page_extracts_public_dc_description_metadata() -> None:
    assert _nature_article_abstract(b'''<html><head>
      <meta name="dc.description" content="Official &amp; complete abstract."/>
      <meta name="description" content="Short teaser."/>
    </head></html>''') == "Official & complete abstract."
    assert _nature_article_document_type(
        b'''<script>window.dataLayer = [{"content":{"category":{"legacy":{
          "webtrendsContentSubGroup":"Matters Arising"}}}}];</script>'''
    ) == "Matters Arising"


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


def test_pmlr_detail_extracts_official_abstract() -> None:
    assert _pmlr_detail(
        b'''<div id="abstract">An <em>official</em> &amp; complete abstract.</div>'''
    ) == {
        "abstract": "An official & complete abstract.",
        "abstract_source": "pmlr_official_detail:abstract",
    }


def test_pmlr_detail_fallback_preserves_primary_pdf_after_bulk_failure() -> None:
    class Transport:
        def fetch_metadata(self, provider, url, **kwargs):
            if "codeload.github.com" in url:
                raise IncompleteRead(b"partial", 100)
            return type(
                "Response",
                (),
                {"body": b'<div id="abstract">Detail abstract.</div>'},
            )()

    entry = SourceEntry(
        "pmlr",
        "v235/audit24a",
        "Auditable Metadata",
        landing_url="https://proceedings.mlr.press/v235/audit24a.html",
        pdf_url="https://proceedings.mlr.press/v235/audit24a/audit24a.pdf",
    )
    result = OfficialStage1FieldHydrator(Transport())._pmlr((entry,))

    assert result.entries[0].abstract == "Detail abstract."
    assert result.entries[0].pdf_url == entry.pdf_url
    assert result.warnings


def test_pmlr_detail_fallback_can_skip_large_bulk_snapshot() -> None:
    class Transport:
        def fetch_metadata(self, provider, url, **kwargs):
            assert "codeload.github.com" not in url
            return type(
                "Response",
                (),
                {"body": b'<div id="abstract">Detail abstract.</div>'},
            )()

    entry = SourceEntry(
        "pmlr",
        "v238/audit24a",
        "Auditable Metadata",
        landing_url="https://proceedings.mlr.press/v238/audit24a.html",
        pdf_url="https://proceedings.mlr.press/v238/audit24a/audit24a.pdf",
    )
    result = OfficialStage1FieldHydrator(Transport())._pmlr(
        (entry,), skip_bulk_snapshot=True
    )

    assert result.entries[0].abstract == "Detail abstract."
    assert result.entries[0].pdf_url == entry.pdf_url


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


def test_jmlr_rss_extracts_abstract_and_public_pdf() -> None:
    body = b'''<rss><channel><item>
      <link>http://jmlr.org/papers/v25/24-0001.html</link>
      <pdf>http://jmlr.org/papers/volume25/24-0001/24-0001.pdf</pdf>
      <description>Official journal abstract.</description>
    </item></channel></rss>'''

    assert _jmlr_rss_records(body) == {
        "v25/24-0001": {
            "abstract": "Official journal abstract.",
            "pdf_url": "https://jmlr.org/papers/volume25/24-0001/24-0001.pdf",
        }
    }


def test_jmlr_detail_extracts_abstract_and_public_pdf() -> None:
    body = b'''<meta name="citation_pdf_url" content="http://jmlr.org/paper.pdf">
      <h3>Abstract</h3><p class="abstract">An &amp; official <em>abstract</em>.</p>'''
    assert _jmlr_detail(body) == {
        "abstract": "An & official abstract.",
        "pdf_url": "https://jmlr.org/paper.pdf",
    }


def test_acl_crossref_page_joins_by_anthology_id() -> None:
    body = b'''{"message":{"items":[
      {"DOI":"10.18653/v1/2024.acl-long.475","abstract":"<jats:p>Registry abstract.</jats:p>"}
    ],"next-cursor":"next"}}'''
    assert _acl_crossref_page(body) == (
        {"2024.acl-long.475": {
            "doi": "10.18653/v1/2024.acl-long.475",
            "abstract": "Registry abstract.",
        }},
        None,
    )

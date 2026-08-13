from __future__ import annotations

from email.message import Message
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from paper_agent.domain import AccessBasis, PublicationVersion
from paper_agent.http_transport import ControlledHTTPTransport, HTTPProviderDelegate
from paper_agent.provider_runtime import ProviderPolicyDenied, ProviderRequestError, ProviderRuntime, ProviderRuntimePolicy
from paper_agent.providers.api import CrawlWindow, VenueDescriptor
from paper_agent.providers.builtin import create_builtin
from paper_agent.venue_transport import (
    VenueOperationResult,
    _html,
    _pmlr_publication_date,
    execute_venue_operation,
    venue_provider_names,
)


FIXTURES = Path(__file__).parent / "fixtures" / "providers"
VENUE_PROVIDERS = (
    "crossref_serial",
    "dblp_toc",
    "neurips_proceedings",
    "pmlr",
    "jmlr_official",
    "acl_anthology",
    "cvf_open_access",
    "ijcai_proceedings",
    "openreview",
    "aaai_ojs",
    "eda_proceedings",
)


class Response:
    def __init__(self, body: bytes, content_type: str) -> None:
        self._body = body
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: object) -> None:
        return None


class FixtureOpener:
    def __init__(self, routes: dict[str, tuple[str, str]]) -> None:
        self.routes = routes
        self.calls = []

    def __call__(self, request, timeout):
        assert ".pdf" not in request.full_url.casefold()
        self.calls.append((request, timeout))
        for marker, (fixture, content_type) in self.routes.items():
            if marker in request.full_url:
                return Response((FIXTURES / fixture).read_bytes(), content_type)
        raise AssertionError(f"unexpected URL: {request.full_url}")


def _runtime(
    *, eda_terms: bool = True, denied_policies: tuple[str, ...] = ()
) -> ProviderRuntime:
    policies = {
            provider: ProviderRuntimePolicy(
                provider,
                cache_ttl_seconds=3600,
                terms_accepted=(
                    eda_terms if provider == "eda_proceedings" else True
                ) and provider not in denied_policies,
                retry_attempts=1,
                jitter_seconds=0,
            )
            for provider in VENUE_PROVIDERS
        }
    policies["eda_proceedings:ieee_xplore"] = ProviderRuntimePolicy(
        "eda_proceedings:ieee_xplore",
        credentials_required=True,
        credential_environment_variables=("IEEE_XPLORE_API_KEY",),
        terms_accepted=eda_terms and "eda_proceedings:ieee_xplore" not in denied_policies,
        retry_attempts=1,
        jitter_seconds=0,
    )
    policies["eda_proceedings:acm_dl"] = ProviderRuntimePolicy(
        "eda_proceedings:acm_dl",
        terms_accepted=eda_terms and "eda_proceedings:acm_dl" not in denied_policies,
        retry_attempts=1,
        jitter_seconds=0,
    )
    for series in ("dac", "iccad"):
        name = f"eda_proceedings:{series}_program"
        policies[name] = ProviderRuntimePolicy(
            name,
            terms_accepted=eda_terms and name not in denied_policies,
            retry_attempts=1,
            jitter_seconds=0,
        )
    return ProviderRuntime(policies)


def _transport(routes: dict[str, tuple[str, str]]) -> tuple[ControlledHTTPTransport, FixtureOpener]:
    opener = FixtureOpener(routes)
    return ControlledHTTPTransport("operator@example.test", opener=opener, runtime=_runtime()), opener


def _official_eda_routes(series: str) -> dict[int, dict[str, str]]:
    if series == "DAC":
        return {
            2024: {
                "route_kind": "dac_linklings_html",
                "url": "https://61dac.conference-program.com/search-program/",
            }
        }
    return {
        2024: {
            "route_kind": "iccad_accepted_html",
            "url": "https://2024.iccad.com/accepted-papers/",
        }
    }


def test_venue_handler_registry_covers_every_conference_primary_provider() -> None:
    assert venue_provider_names() == tuple(sorted(VENUE_PROVIDERS))


def test_dblp_toc_maps_complete_year_with_doi_and_census() -> None:
    transport, opener = _transport({
        "dblp.org/db/conf/dac/dac2024.xml": (
            "http-venue-dblp-dac.xml",
            "application/xml",
        ),
    })
    first = transport(
        "dblp_toc",
        "discover",
        {
            "toc_series": "dac",
            "exclude_titles": ["Frontmatter"],
            "venue_id": "dac",
            "year": 2024,
            "page_size": 1,
        },
    )

    assert first["next_cursor"] is None
    assert first["census"] == {
        "expected_total": 1,
        "parser_raw_records": 2,
        "parser_rejected_records": 0,
        "parser_excluded_records": 1,
    }
    assert first["entries"][0]["external_id"] == "conf/dac/Test24"
    assert first["entries"][0]["doi"] == "10.1145/fixture.dac"
    assert first["entries"][0]["authors"] == ["Ada Lovelace"]
    assert len(opener.calls) == 1


def test_dblp_toc_falls_back_between_approved_hosts() -> None:
    body = (FIXTURES / "http-venue-dblp-iclr.xml").read_bytes()
    urls: list[str] = []

    class Body:
        def __init__(self) -> None:
            self.body = body

    def fetch(url: str, api_version: str):
        urls.append(url)
        assert api_version == "dblp-toc-xml-v1"
        if url.startswith("https://dblp.org/"):
            raise OSError("fixture primary connection closed")
        return Body()

    result = execute_venue_operation(
        "dblp_toc",
        "discover",
        {
            "toc_series": "iclr",
            "toc_base_urls": ["https://dblp.org", "https://dblp.uni-trier.de"],
            "venue_id": "iclr",
            "year": 2024,
        },
        fetch,
    )

    assert [entry["external_id"] for entry in result.payload["entries"]] == [
        "conf/iclr/Test24"
    ]
    assert urls == [
        "https://dblp.org/db/conf/iclr/iclr2024.xml",
        "https://dblp.uni-trier.de/db/conf/iclr/iclr2024.xml",
    ]


def test_crossref_serial_cursor_tracks_consumed_total_and_stops_exactly() -> None:
    pages = [
        {
            "message": {
                "total-results": 2,
                "next-cursor": "registry-cursor-2",
                "items": [
                    {
                        "DOI": "10.1000/first",
                        "title": ["First registry work"],
                        "published": {"date-parts": [[2024, 1, 2]]},
                        "container-title": ["Fixture Journal"],
                        "type": "journal-article",
                        "link": [{
                            "URL": "https://publisher.example/first.pdf",
                            "content-type": "application/pdf",
                        }],
                    }
                ],
            }
        },
        {
            "message": {
                "total-results": 2,
                "next-cursor": "registry-cursor-3",
                "items": [
                    {
                        "DOI": "10.1000/second",
                        "title": ["Second registry work"],
                        "published-online": {"date-parts": [[2024, 12, 31]]},
                        "container-title": ["Fixture Journal"],
                        "type": "journal-article",
                    }
                ],
            }
        },
    ]
    urls: list[str] = []

    class Body:
        def __init__(self, payload: dict[str, object]) -> None:
            self.body = json.dumps(payload).encode()
            self.content_type = "application/json"

    request_keys: list[str | None] = []

    def fetch(
        url: str,
        api_version: str,
        policy_provider: str | None = None,
        request_key: str | None = None,
    ):
        urls.append(url)
        request_keys.append(request_key)
        assert api_version == "crossref-serial-rest-v1"
        assert policy_provider is None
        return Body(pages[len(urls) - 1])

    parameters = {
        "year": 2024,
        "registry_issn": "1234-5678",
        "issns": ["1234-5678"],
        "page_size": 1,
    }
    first = execute_venue_operation("crossref_serial", "discover", parameters, fetch)
    second = execute_venue_operation(
        "crossref_serial",
        "discover",
        {**parameters, "cursor": first.payload["next_cursor"]},
        fetch,
    )

    assert first.payload["next_cursor"] == "1:registry-cursor-2"
    assert second.payload["next_cursor"] is None
    assert [page.payload["entries"][0]["external_id"] for page in (first, second)] == [
        "10.1000/first",
        "10.1000/second",
    ]
    assert first.payload["entries"][0]["pdf_url"] == (
        "https://publisher.example/first.pdf"
    )
    assert "cursor=%2A" in urls[0]
    assert "cursor=registry-cursor-2" in urls[1]
    assert request_keys == ["offset:0", "offset:1"]
    assert first.payload["census"] == second.payload["census"] == {
        "expected_total": 2,
        "parser_raw_records": 2,
        "parser_rejected_records": 0,
        "parser_excluded_records": 0,
    }


def test_crossref_serial_request_key_bypasses_same_url_cache() -> None:
    payloads = [
        {
            "message": {
                "total-results": 2,
                "next-cursor": "stable-server-cursor",
                "items": [
                    {
                        "DOI": "10.1000/page-one",
                        "title": ["Page one"],
                        "published": {"date-parts": [[2024, 1, 1]]},
                    }
                ],
            }
        },
        {
            "message": {
                "total-results": 2,
                "next-cursor": "stable-server-cursor",
                "items": [
                    {
                        "DOI": "10.1000/page-two",
                        "title": ["Page two"],
                        "published": {"date-parts": [[2024, 2, 1]]},
                    }
                ],
            }
        },
    ]

    class AdvancingOpener:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, request, timeout):
            assert "mailto=operator%40example.test" in request.full_url
            payload = payloads[self.calls]
            self.calls += 1
            return Response(json.dumps(payload).encode(), "application/json")

    opener = AdvancingOpener()
    transport = ControlledHTTPTransport(
        "operator@example.test",
        opener=opener,
        runtime=_runtime(),
    )
    base = {
        "year": 2024,
        "registry_issn": "1234-5678",
        "issns": ["1234-5678"],
        "page_size": 1,
    }
    first = transport("crossref_serial", "discover", base)
    second = transport(
        "crossref_serial",
        "discover",
        {**base, "cursor": first["next_cursor"]},
    )

    assert opener.calls == 2
    assert first["entries"][0]["external_id"] == "10.1000/page-one"
    assert second["entries"][0]["external_id"] == "10.1000/page-two"
    assert first["next_cursor"] == "1:stable-server-cursor"
    assert second["next_cursor"] is None
    assert transport.request_audit[0]["request_key"] == "offset:0"
    assert transport.request_audit[1]["request_key"] == "offset:1"


def test_neurips_uses_year_page_maps_official_ids_and_paginates_cached_html() -> None:
    transport, opener = _transport({
        "proceedings.neurips.cc/paper_files/paper/2024": ("http-venue-neurips.html", "text/html"),
    })
    descriptor = VenueDescriptor(
        1, "neurips", "neurips_proceedings", "neurips_proceedings", {"series": "NeurIPS", "page_size": 1}
    )
    adapter = create_builtin("neurips_proceedings", transport)

    window = CrawlWindow(year=2024)
    first = adapter.discover(descriptor, window)
    second = adapter.discover(descriptor, window, first.next_cursor)

    assert opener.calls[0][0].full_url == "https://proceedings.neurips.cc/paper_files/paper/2024"
    assert len(opener.calls) == 1
    assert first.next_cursor == "1" and second.next_cursor == "2"
    assert first.entries[0].external_id == "NeurIPS-2024-abc123"
    assert first.entries[0].authors == ("Ada Lovelace", "Grace Hopper")
    assert first.entries[0].pdf_url == (
        "https://proceedings.neurips.cc/paper_files/paper/2024/file/"
        "abc123-Paper-Conference.pdf"
    )
    assert first.entries[0].publication_version is PublicationVersion.PUBLISHED
    assert first.entries[0].host_type == "official"
    assert first.entries[0].access_basis is AccessBasis.PUBLIC_READ_ONLY
    assert first.entries[0].metadata["language"] == "en"
    assert first.entries[0].metadata["document_type"] == "proceedings-article"
    assert second.entries[0].title == "Second Paper"
    assert second.entries[0].landing_url.endswith("def456-Abstract-Conference.html")

    third = adapter.discover(descriptor, window, second.next_cursor)
    assert third.next_cursor is None
    assert third.entries[0].title == "Dataset Track Paper"
    assert third.entries[0].pdf_url.endswith(
        "data789-Paper-Datasets_and_Benchmarks_Track.pdf"
    )


def test_neurips_legacy_page_accepts_unclassified_items_and_plain_abstract_suffix() -> None:
    transport, _ = _transport({
        "proceedings.neurips.cc/paper_files/paper/2020": ("http-venue-neurips-legacy.html", "text/html"),
    })
    descriptor = VenueDescriptor(
        1, "neurips", "neurips_proceedings", "neurips_proceedings", {"series": "NeurIPS"}
    )
    batch = create_builtin("neurips_proceedings", transport).discover(descriptor, CrawlWindow(year=2020))

    assert batch.entries[0].external_id == "NeurIPS-2020-legacy123"
    assert batch.entries[0].landing_url.endswith("legacy123-Abstract.html")
    assert batch.entries[0].pdf_url == (
        "https://proceedings.neurips.cc/paper_files/paper/2020/file/legacy123-Paper.pdf"
    )
    assert batch.entries[0].metadata["language"] == "en"
    assert batch.entries[0].metadata["document_type"] == "proceedings-article"


def test_pmlr_resolves_exact_icml_volume_then_maps_volume_page_without_pdf_fetch() -> None:
    transport, opener = _transport({
        "https://proceedings.mlr.press/v235/": ("http-venue-pmlr-volume.html", "text/html"),
        "https://proceedings.mlr.press/": ("http-venue-pmlr-index.html", "text/html"),
    })
    descriptor = VenueDescriptor(
        1, "icml", "pmlr", "pmlr", {"series": "ICML", "volume_resolution": "official_link", "page_size": 1}
    )
    batch = create_builtin("pmlr", transport).discover(descriptor, CrawlWindow(year=2024, volume="v235"))

    assert [call[0].full_url for call in opener.calls] == [
        "https://proceedings.mlr.press/",
        "https://proceedings.mlr.press/v235/",
    ]
    assert batch.next_cursor == "1"
    assert batch.entries[0].external_id == "v235/lovelace24a"
    assert batch.entries[0].title == "Reliable Small Models"
    assert batch.entries[0].publication_date == "2024-07-08"
    assert batch.entries[0].metadata["volume"] == "235"
    assert [record["api_version"] for record in batch.request_audit] == [
        "pmlr-index-html-v1",
        "pmlr-volume-html-v1",
    ]


def test_pmlr_colt_excludes_declared_preface_and_reconciles_census() -> None:
    index = b"""<html><body><ul><li><a href='v247'><b>Volume 247</b></a>
    Proceedings of COLT 2024</li></ul></body></html>"""
    volume = b"""<html><head><meta name='description' content='Held in Edmonton on 30 June 2024.'></head><body>
    <div class='paper'><p class='title'>Conference on Learning Theory 2024: Preface</p><p class='authors'>Chairs</p><a href='/v247/preface.html'>abs</a></div>
    <div class='paper'><p class='title'>A Research Paper</p><p class='authors'>Ada Lovelace</p><a href='/v247/paper.html'>abs</a></div>
    </body></html>"""

    class Response:
        content_type = "text/html"

        def __init__(self, body: bytes) -> None:
            self.body = body

    def fetch(url: str, api_version: str, policy_provider: str | None = None):
        return Response(volume if "/v247/" in url else index)

    resolved = execute_venue_operation(
        "pmlr", "resolve_volume", {"series": "COLT", "year": 2024}, fetch
    )
    assert resolved.payload["official_url"] == "https://proceedings.mlr.press/v247/"
    result = execute_venue_operation(
        "pmlr",
        "discover",
        {
            "series": "COLT",
            "year": 2024,
            "volume_id": "v247",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
            "exclude_title_patterns": [r"\bPreface\b"],
        },
        fetch,
    )
    assert [entry["external_id"] for entry in result.payload["entries"]] == ["v247/paper"]
    assert result.payload["census"] == {
        "expected_total": 1,
        "parser_raw_records": 2,
        "parser_rejected_records": 0,
        "parser_excluded_records": 1,
    }


def test_pmlr_corl_uses_held_date_when_published_next_year() -> None:
    index = b"""<html><body><ul><li><a href='v270'><b>Volume 270</b></a>
    Proceedings of CoRL 2024</li></ul></body></html>"""
    volume = b"""<html><head><meta name='description' content='Proceedings of CoRL Held in Munich on 06-09 November 2024. Published on 10 January 2025.'></head><body>
    <div class='paper'><p class='title'>A Robot Learning Paper</p><p class='authors'>Ada Lovelace</p><a href='/v270/paper.html'>abs</a></div>
    </body></html>"""

    class Response:
        content_type = "text/html"

        def __init__(self, body: bytes) -> None:
            self.body = body

    def fetch(url: str, api_version: str, policy_provider: str | None = None):
        return Response(volume if "/v270/" in url else index)

    resolved = execute_venue_operation(
        "pmlr", "resolve_volume", {"series": "CORL", "year": 2024}, fetch
    )
    assert resolved.payload["official_url"].endswith("/v270/")
    result = execute_venue_operation(
        "pmlr",
        "discover",
        {
            "series": "CORL",
            "year": 2024,
            "volume_id": "v270",
            "date_from": "2024-01-01",
            "date_to": "2024-12-31",
        },
        fetch,
    )
    assert result.payload["entries"][0]["publication_date"] == "2024-11-06"


def test_pmlr_corl_parses_cross_month_conference_range() -> None:
    root = _html(
        b"""<html><head><meta name='description' content='Proceedings of CoRL on 30 October to 01 November 2019 Published on 12 May 2020.'></head></html>""",
        "pmlr",
    )
    assert _pmlr_publication_date(root, 2019, conference_year=2019) == "2019-10-30"


def test_uai_legacy_auai_page_transcodes_and_maps_official_pdf() -> None:
    html = """<!doctype html><html><body><table>
    <tr><td><b>ID: 7</b><a href='proceedings/papers/7.pdf'>pdf</a></td>
    <td><div class='collapse'><b>Se\u00f1or Legacy Paper</b><div>Legacy abstract.</div></div>
    <i>Ada Lovelace, Example University</i></td></tr>
    </table></body></html>""".encode("latin-1")
    calls: list[tuple[str, str, str | None]] = []

    class LegacyResponse:
        body = html
        content_type = "text/html; charset=iso-8859-1"

    def fetch(url: str, api_version: str, policy_provider: str | None = None):
        calls.append((url, api_version, policy_provider))
        return LegacyResponse()

    result = execute_venue_operation(
        "pmlr",
        "discover",
        {
            "series": "UAI",
            "year": 2016,
            "volume_id": "v0",
            "date_from": "2016-01-01",
            "date_to": "2016-12-31",
        },
        fetch,
    )

    assert calls == [
        (
            "https://www.auai.org/uai2016/proceedings.php",
            "auai-proceedings-html-v1",
            "pmlr:auai_press",
        )
    ]
    assert result.payload["entries"] == [
        {
            "external_id": "uai-2016-7",
            "title": "Se\u00f1or Legacy Paper",
            "authors": ["Ada Lovelace, Example University"],
            "abstract": "Legacy abstract.",
            "publication_date": "2016",
            "year": 2016,
            "venue": "UAI 2016",
            "landing_url": "https://www.auai.org/uai2016/proceedings/papers/7.pdf",
            "pdf_url": "https://www.auai.org/uai2016/proceedings/papers/7.pdf",
            "publication_version": "published",
            "host_type": "official",
            "access_basis": "public_read_only",
            "document_type": "proceedings-article",
            "upstream": "auai_press",
        }
    ]
    assert result.payload["census"]["expected_total"] == 1
    assert result.payload["warnings"]


def test_pmlr_uses_uai_held_year_when_volume_was_published_later() -> None:
    root = _html(
        b"""<html><head><meta name='description' content='Proceedings of UAI Held in Tel Aviv on 22-25 July 2019 Published on 06 August 2020.'></head></html>""",
        "pmlr",
    )
    assert _pmlr_publication_date(root, 2019, conference_year=2019) == "2019-07-22"


def test_jmlr_resolves_every_volume_covering_year_then_filters_paper_year() -> None:
    index = b"""<html><body>
    <p><a href='v18'>Volume 18</a> (February 2017 - August 2018)</p>
    <p><a href='v19'>Volume 19</a> (August 2018 - December 2018)</p>
    </body></html>"""
    volume18 = b"""<html><body>
    <dl><dt>Old Paper</dt><dd><i>Old Author</i>; (1):1-2, 2017.
    <a href='/papers/v18/old.html'>abs</a><a href='/papers/volume18/old/old.pdf'>pdf</a></dd></dl>
    <dl><dt>First 2018 Paper</dt><dd><i>Ada Lovelace, Grace Hopper</i>; (2):1-9, 2018.
    <a href='/papers/v18/first.html'>abs</a><a href='/papers/volume18/first/first.pdf'>pdf</a></dd></dl>
    </body></html>"""
    volume19 = b"""<html><body>
    <dl><dt>Second 2018 Paper</dt><dd><i>Alan Turing</i>; (1):1-5, 2018.
    <a href='/papers/v19/second.html'>abs</a><a href='/papers/volume19/second/second.pdf'>pdf</a></dd></dl>
    </body></html>"""
    bodies = {
        "https://www.jmlr.org/papers/": index,
        "https://www.jmlr.org/papers/v18/": volume18,
        "https://www.jmlr.org/papers/v19/": volume19,
    }
    calls: list[str] = []

    class Response:
        content_type = "text/html"

        def __init__(self, body: bytes) -> None:
            self.body = body

    def fetch(url: str, api_version: str, policy_provider: str | None = None):
        calls.append(url)
        return Response(bodies[url])

    result = execute_venue_operation(
        "jmlr_official",
        "discover",
        {"year": 2018, "date_from": "2018-01-01", "date_to": "2018-12-31"},
        fetch,
    )

    assert calls == list(bodies)
    assert [entry["external_id"] for entry in result.payload["entries"]] == [
        "v18/first",
        "v19/second",
    ]
    assert result.payload["entries"][0]["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert result.payload["entries"][0]["pdf_url"].endswith("/first/first.pdf")
    assert result.payload["census"] == {
        "expected_total": 2,
        "parser_raw_records": 3,
        "parser_rejected_records": 0,
        "parser_excluded_records": 1,
    }


def test_acl_uses_frozen_official_xml_and_honors_collection_and_date_window() -> None:
    snapshot = "1941968b51805719b418a0b0919e335662cdd172"
    transport, opener = _transport({
        f"/{snapshot}/data/xml/2024.acl.xml": ("http-venue-acl.xml", "application/xml"),
    })
    descriptor = VenueDescriptor(
        1,
        "acl",
        "acl_anthology",
        "acl_anthology",
        {"snapshot_version": snapshot, "collections": ["main"]},
    )
    batch = create_builtin("acl_anthology", transport).discover(
        descriptor, CrawlWindow(year=2024, date_from="2024-08-01", date_to="2024-08-31")
    )

    assert opener.calls[0][0].full_url == (
        f"https://raw.githubusercontent.com/acl-org/acl-anthology/{snapshot}/data/xml/2024.acl.xml"
    )
    assert [entry.external_id for entry in batch.entries] == ["2024.acl-long.1"]
    assert batch.entries[0].title == "Structured LLM Reasoning"
    assert batch.entries[0].abstract == "ACL abstract."
    assert batch.entries[0].metadata["snapshot_version"] == snapshot


def test_acl_resolves_findings_and_workshops_from_pinned_event_volume_mapping() -> None:
    snapshot = "1941968b51805719b418a0b0919e335662cdd172"
    transport, opener = _transport({
        "/data/xml/2024.findings.xml": ("http-venue-acl-findings.xml", "application/xml"),
        "/data/xml/2024.nlpws.xml": ("http-venue-acl-workshop.xml", "application/xml"),
        "/data/xml/2024.acl.xml": ("http-venue-acl.xml", "application/xml"),
    })
    descriptor = VenueDescriptor(
        1,
        "acl",
        "acl_anthology",
        "acl_anthology",
        {"snapshot_version": snapshot, "collections": ["main", "findings", "workshop"]},
    )
    batch = create_builtin("acl_anthology", transport).discover(descriptor, CrawlWindow(year=2024))

    assert {entry.metadata["collection"] for entry in batch.entries} == {"main", "findings", "workshop"}
    assert [urlsplit(call[0].full_url).path.rsplit("/", 1)[-1] for call in opener.calls] == [
        "2024.acl.xml",
        "2024.findings.xml",
        "2024.nlpws.xml",
    ]
    assert len(batch.request_audit) == 3
    assert all(record["response_sha256"] for record in batch.request_audit)


def test_cvf_main_route_preserves_track_and_month_resolution() -> None:
    transport, opener = _transport({
        "openaccess.thecvf.com/CVPR2024?day=all": ("http-venue-cvf.html", "text/html"),
    })
    descriptor = VenueDescriptor(
        1,
        "cvpr",
        "cvf_open_access",
        "cvf_open_access",
        {"series": "CVPR", "track": "main", "proceedings_only": True, "exclude_workshops": True},
    )
    batch = create_builtin("cvf_open_access", transport).discover(descriptor, CrawlWindow(year=2024))

    assert opener.calls[0][0].full_url == "https://openaccess.thecvf.com/CVPR2024?day=all"
    assert len(batch.entries) == 2
    assert batch.entries[0].authors == ("Ada Lovelace", "Grace Hopper")
    assert batch.entries[0].publication_date == "2024-06"
    assert batch.entries[0].metadata["cvf_track"] == "main"
    assert batch.entries[0].pdf_url == (
        "https://openaccess.thecvf.com/content/CVPR2024/papers/"
        "Lovelace_Vision_Systems_CVPR_2024_paper.pdf"
    )


def test_cvf_falls_back_to_official_day_pages_when_legacy_all_query_is_broken() -> None:
    transport, opener = _transport({
        "CVPR2018?day=all": ("http-venue-cvf-empty.html", "text/html"),
        "CVPR2018.py?day=2018-06-19": ("http-venue-cvf.html", "text/html"),
        "CVPR2018.py?day=2018-06-20": ("http-venue-cvf.html", "text/html"),
        "CVPR2018.py": ("http-venue-cvf-menu.html", "text/html"),
    })
    descriptor = VenueDescriptor(
        1,
        "cvpr",
        "cvf_open_access",
        "cvf_open_access",
        {"series": "CVPR", "track": "main", "proceedings_only": True, "exclude_workshops": True},
    )

    batch = create_builtin("cvf_open_access", transport).discover(
        descriptor, CrawlWindow(year=2018)
    )

    assert len(batch.entries) == 2
    assert len(opener.calls) == 4
    assert any("CVPR2018.py?day=2018-06-19" in call[0].full_url for call in opener.calls)


def test_ijcai_uses_details_links_and_never_pdf_links() -> None:
    transport, opener = _transport({
        "www.ijcai.org/proceedings/2024/": ("http-venue-ijcai.html", "text/html"),
    })
    descriptor = VenueDescriptor(1, "ijcai", "ijcai_proceedings", "ijcai_proceedings", {"series": "IJCAI"})
    batch = create_builtin("ijcai_proceedings", transport).discover(descriptor, CrawlWindow(year=2024))

    assert len(opener.calls) == 1
    assert [entry.external_id for entry in batch.entries] == ["IJCAI-2024-1", "IJCAI-2024-2"]
    assert batch.entries[0].landing_url == "https://www.ijcai.org/proceedings/2024/1"


def test_ijcai_legacy_template_excludes_frontmatter_and_emits_direct_pdf() -> None:
    transport, _ = _transport({
        "www.ijcai.org/proceedings/2016/": (
            "http-venue-ijcai-legacy.html",
            "text/html",
        ),
    })
    descriptor = VenueDescriptor(
        1, "ijcai", "ijcai_proceedings", "ijcai_proceedings", {"series": "IJCAI"}
    )

    batch = create_builtin("ijcai_proceedings", transport).discover(
        descriptor, CrawlWindow(year=2016)
    )

    assert [entry.external_id for entry in batch.entries] == ["IJCAI-2016-008", "IJCAI-2016-009"]
    assert batch.entries[0].pdf_url == "https://www.ijcai.org/Proceedings/16/Papers/008.pdf"


def test_ijcai_duplicate_official_listing_is_counted_as_explicit_exclusion() -> None:
    transport, _ = _transport({
        "www.ijcai.org/proceedings/2021/": (
            "http-venue-ijcai-duplicate.html",
            "text/html",
        ),
    })
    descriptor = VenueDescriptor(
        1, "ijcai", "ijcai_proceedings", "ijcai_proceedings", {"series": "IJCAI"}
    )

    payload = transport(
        "ijcai_proceedings",
        "discover",
        {"series": "IJCAI", "year": 2021, "venue_id": "ijcai"},
    )

    assert len(payload["entries"]) == 1
    assert payload["census"] == {
        "expected_total": 1,
        "parser_raw_records": 2,
        "parser_rejected_records": 0,
        "parser_excluded_records": 1,
    }


def test_openreview_resolves_v2_venueid_and_maps_value_wrappers_and_cursor() -> None:
    transport, opener = _transport({
        "api2.openreview.net/groups": ("http-venue-openreview-group.json", "application/json"),
        "api2.openreview.net/notes": ("http-venue-openreview.json", "application/json"),
    })
    descriptor = VenueDescriptor(
        1,
        "iclr",
        "openreview",
        "openreview",
        {"venue_group": "ICLR.cc", "accepted_decision_required": True, "page_size": 2},
    )
    batch = create_builtin("openreview", transport).discover(
        descriptor, CrawlWindow(year=2024, date_from="2024-01-01", date_to="2024-12-31")
    )

    assert parse_qs(urlsplit(opener.calls[0][0].full_url).query) == {
        "id": ["ICLR.cc/2024/Conference"]
    }
    query = parse_qs(urlsplit(opener.calls[1][0].full_url).query)
    assert urlsplit(opener.calls[1][0].full_url).netloc == "api2.openreview.net"
    assert query == {"venueid": ["ICLR.cc/2024/Conference"], "limit": ["2"], "offset": ["0"]}
    assert batch.next_cursor == "2"
    assert [entry.external_id for entry in batch.entries] == ["openreview-note-1", "openreview-note-2"]
    assert batch.entries[0].authors == ("Ada Lovelace", "Grace Hopper")
    assert batch.entries[0].abstract == "OpenReview abstract."
    assert [record["api_version"] for record in batch.request_audit] == [
        "openreview-api-v2-group",
        "openreview-api-v2",
    ]


def test_openreview_dynamic_resolution_rejects_legacy_group_without_exact_invitation() -> None:
    class LegacyGroupOpener:
        def __call__(self, request, timeout):
            return Response(
                b'{"groups":[{"id":"ICLR.cc/2024/Conference","domain":null,"content":{}}]}',
                "application/json",
            )

    transport = ControlledHTTPTransport(
        "operator@example.test", opener=LegacyGroupOpener(), runtime=_runtime()
    )
    with pytest.raises(ProviderRequestError, match="freeze invitation"):
        transport(
            "openreview",
            "resolve_invitation",
            {"venue_group": "ICLR.cc", "year": 2024},
        )


def test_aaai_traverses_archive_and_issue_then_paginates_articles_from_cache() -> None:
    transport, opener = _transport({
        "/issue/archive": ("http-venue-aaai-archive.html", "text/html"),
        "/issue/view/701": ("http-venue-aaai-issue.html", "text/html"),
    })
    descriptor = VenueDescriptor(
        1, "aaai", "aaai_ojs", "aaai_ojs", {"journal": "AAAI", "traverse_all_issues": True, "page_size": 1}
    )
    adapter = create_builtin("aaai_ojs", transport)
    window = CrawlWindow(year=2024, volume="38", issue="1")
    first = adapter.discover(descriptor, window)
    second = adapter.discover(descriptor, window, first.next_cursor)

    assert [call[0].full_url for call in opener.calls] == [
        "https://ojs.aaai.org/index.php/AAAI/issue/archive",
        "https://ojs.aaai.org/index.php/AAAI/issue/view/701",
    ]
    assert first.next_cursor == "1" and second.next_cursor is None
    assert first.entries[0].external_id == "31001"
    assert first.entries[0].publication_date == "2024-03-24"
    assert first.entries[0].pdf_url == (
        "https://ojs.aaai.org/index.php/AAAI/article/download/31001/33001"
    )
    assert first.entries[0].metadata["ojs_issue_id"] == "701"
    assert first.entries[0].metadata["volume"] == "38"
    assert first.entries[0].metadata["issue"] == "1"


def test_aaai_collects_same_year_issues_across_archive_pages() -> None:
    transport, opener = _transport({
        "/issue/archive?page=2": ("http-venue-aaai-archive-page-2.html", "text/html"),
        "/issue/archive": ("http-venue-aaai-archive-page-1.html", "text/html"),
        "/issue/view/701": ("http-venue-aaai-issue.html", "text/html"),
        "/issue/view/702": ("http-venue-aaai-issue-702.html", "text/html"),
    })
    descriptor = VenueDescriptor(
        1, "aaai", "aaai_ojs", "aaai_ojs", {"journal": "AAAI", "traverse_all_issues": True}
    )
    batch = create_builtin("aaai_ojs", transport).discover(descriptor, CrawlWindow(year=2024, volume="38"))

    assert {entry.metadata["ojs_issue_id"] for entry in batch.entries} == {"701", "702"}
    assert [call[0].full_url for call in opener.calls[:2]] == [
        "https://ojs.aaai.org/index.php/AAAI/issue/archive",
        "https://ojs.aaai.org/index.php/AAAI/issue/archive?page=2",
    ]
    assert len(batch.request_audit) == 4


@pytest.mark.parametrize(
    ("series", "marker", "fixture", "expected_id"),
    [
        ("DAC", "61dac.conference-program.com/search-program/", "http-venue-dac.html", "DAC-2024-RESEARCH1860"),
        ("ICCAD", "2024.iccad.com/accepted-papers/", "http-venue-iccad.html", "ICCAD-2024-510"),
    ],
)
def test_eda_official_pages_preserve_program_provenance_without_claiming_an_upstream(
    series: str, marker: str, fixture: str, expected_id: str
) -> None:
    opener = FixtureOpener({marker: (fixture, "text/html")})
    transport = ControlledHTTPTransport(
        "operator@example.test",
        opener=opener,
        runtime=_runtime(denied_policies=("eda_proceedings:acm_dl",)),
    )
    proceedings_doi = "10.1145/3649329" if series == "DAC" else "10.1145/3676536"
    descriptor = VenueDescriptor(
        1,
        series.casefold(),
        "eda_proceedings",
        "eda_proceedings",
        {
            "series": series,
            "upstreams": ["ieee_xplore", "acm_dl"],
            "resolve_platforms_by_year": True,
            "deduplicate_by": "doi",
            "official_routes_by_year": _official_eda_routes(series),
            "upstream_routes_by_year": {
                2024: {
                    "acm_dl": {"proceedings_doi": proceedings_doi, "required": True}
                }
            },
        },
    )
    batch = create_builtin("eda_proceedings", transport).discover(descriptor, CrawlWindow(year=2024))

    assert marker in opener.calls[0][0].full_url
    assert batch.entries[0].external_id == expected_id
    assert batch.status.value == "partial"
    assert "acm_dl: terms are not accepted" in (batch.error or "")
    assert "upstream" not in batch.entries[0].metadata
    assert batch.entries[0].metadata["provenance"] == {
        "source": "dac_linklings_html" if series == "DAC" else "iccad_accepted_html",
        "url": opener.calls[0][0].full_url,
    }
    assert batch.entries[0].metadata["upstream_resolution"] == {
        "status": "unresolved",
        "candidates": ["ieee_xplore", "acm_dl"],
        "doi_candidates": [],
    }
    if series == "DAC":
        assert batch.entries[0].authors == ("Ada Lovelace", "Grace Hopper")
        assert batch.entries[0].abstract == "DAC abstract."


def test_eda_resolves_two_independent_upstreams_with_credential_and_doi_metadata() -> None:
    opener = FixtureOpener({
        "61dac.conference-program.com/search-program/": ("http-venue-dac.html", "text/html"),
        "ieeexploreapi.ieee.org/api/v1/search/articles": ("http-venue-eda-ieee.json", "application/json"),
        "dl.acm.org/doi/proceedings/10.1145/3649329": ("http-venue-eda-acm.html", "text/html"),
    })
    transport = ControlledHTTPTransport(
        "operator@example.test",
        opener=opener,
        runtime=_runtime(),
        environment={"IEEE_XPLORE_API_KEY": "ieee-secret"},
    )
    descriptor = VenueDescriptor(
        1,
        "dac",
        "eda_proceedings",
        "eda_proceedings",
        {
            "series": "DAC",
            "upstreams": ["ieee_xplore", "acm_dl"],
            "resolve_platforms_by_year": True,
            "deduplicate_by": "doi",
            "official_routes_by_year": _official_eda_routes("DAC"),
            "upstream_routes_by_year": {
                2024: {
                    "ieee_xplore": {"publication_number": 11293840, "required": False},
                    "acm_dl": {"proceedings_doi": "10.1145/3649329", "required": True},
                }
            },
        },
    )
    batch = create_builtin("eda_proceedings", transport).discover(descriptor, CrawlWindow(year=2024))

    assert batch.status.value == "success" and batch.error is None
    assert {entry.external_id for entry in batch.entries} == {
        "DAC-2024-RESEARCH1860",
        "DAC-2024-RESEARCH1861",
    }
    assert {entry.doi for entry in batch.entries} == {
        "10.1145/3649329.test0001",
        "10.1145/3649329.3650002",
    }
    assert {tuple(entry.metadata["upstream_resolution"]["sources"]) for entry in batch.entries} == {
        ("acm_dl",),
    }
    assert len(batch.request_audit) == 3
    assert {record["api_version"] for record in batch.request_audit} == {
        "dac-linklings-html-v1",
        "eda-ieee_xplore-metadata-v1",
        "eda-acm_dl-metadata-v1",
    }
    ieee_request = next(call[0] for call in opener.calls if "ieeexploreapi" in call[0].full_url)
    ieee_query = parse_qs(urlsplit(ieee_request.full_url).query)
    assert ieee_query["apikey"] == ["ieee-secret"]
    assert ieee_query["publication_number"] == ["11293840"]
    assert "publication_id" not in ieee_query and "publication_title" not in ieee_query


def test_eda_ieee_official_catalog_reuses_one_request_and_resolves_without_doi() -> None:
    opener = FixtureOpener({
        "ieeexploreapi.ieee.org/api/v1/search/articles": (
            "http-venue-eda-ieee.json",
            "application/json",
        ),
    })
    transport = ControlledHTTPTransport(
        "operator@example.test",
        opener=opener,
        runtime=_runtime(),
        environment={"IEEE_XPLORE_API_KEY": "ieee-secret"},
    )
    evidence_url = "https://ieeexplore.ieee.org/xpl/conhome/11293840/proceeding"

    payload = transport(
        "eda_proceedings",
        "discover",
        {
            "series": "DAC",
            "upstreams": ["ieee_xplore", "acm_dl"],
            "resolve_platforms_by_year": True,
            "deduplicate_by": "doi",
            "year": 2024,
            "official_routes_by_year": {
                2024: {
                    "route_kind": "ieee_xplore_publication",
                    "publication_number": 11293840,
                    "evidence_url": evidence_url,
                }
            },
            "upstream_routes_by_year": {
                2024: {
                    "ieee_xplore": {"publication_number": 11293840, "required": True}
                }
            },
        },
    )

    assert payload["status"] == "success"
    assert payload["incomplete_reasons"] == []
    assert payload["warnings"] == ["acm_dl: no frozen route for 2024"]
    assert payload["unavailable_upstreams"] == ["acm_dl"]
    assert len(opener.calls) == 1
    query = parse_qs(urlsplit(opener.calls[0][0].full_url).query)
    assert query["publication_number"] == ["11293840"]
    assert "publication_id" not in query and "publication_title" not in query
    entry = payload["entries"][0]
    assert entry["external_id"] == "DAC-2024-IEEE-107001"
    assert entry["doi"] is None
    assert entry["official_catalog_source"] == "ieee_xplore"
    assert entry["evidence_url"] == evidence_url
    assert entry["provenance"] == {
        "source": "ieee_xplore_publication",
        "url": evidence_url,
    }
    assert entry["upstream_resolution"] == {
        "status": "resolved_without_doi",
        "sources": ["ieee_xplore"],
        "doi_candidates": [],
    }


@pytest.mark.parametrize(
    ("ieee_required", "expected_status"),
    [(False, "success"), (True, "partial")],
)
def test_eda_only_required_upstream_failure_marks_batch_partial(
    ieee_required: bool, expected_status: str
) -> None:
    opener = FixtureOpener({
        "61dac.conference-program.com/search-program/": ("http-venue-dac.html", "text/html"),
        "dl.acm.org/doi/proceedings/10.1145/3649329": (
            "http-venue-eda-acm.html",
            "text/html",
        ),
    })
    transport = ControlledHTTPTransport(
        "operator@example.test",
        opener=opener,
        runtime=_runtime(denied_policies=("eda_proceedings:ieee_xplore",)),
    )

    payload = transport(
        "eda_proceedings",
        "discover",
        {
            "series": "DAC",
            "upstreams": ["ieee_xplore", "acm_dl"],
            "resolve_platforms_by_year": True,
            "deduplicate_by": "doi",
            "year": 2024,
            "official_routes_by_year": _official_eda_routes("DAC"),
            "upstream_routes_by_year": {
                2024: {
                    "ieee_xplore": {
                        "publication_number": 11293840,
                        "required": ieee_required,
                    },
                    "acm_dl": {
                        "proceedings_doi": "10.1145/3649329",
                        "required": not ieee_required,
                    },
                }
            },
        },
    )

    assert payload["status"] == expected_status
    assert payload["unavailable_upstreams"] == ["ieee_xplore"]
    denial = "ieee_xplore: eda_proceedings:ieee_xplore: terms are not accepted"
    if ieee_required:
        assert payload["incomplete_reasons"] == [denial]
        assert payload["warnings"] == []
    else:
        assert payload["incomplete_reasons"] == []
        assert payload["warnings"] == [denial]


def test_eda_fails_closed_on_conflicting_upstream_dois_and_preserves_official_fields() -> None:
    opener = FixtureOpener({
        "61dac.conference-program.com/search-program/": ("http-venue-dac.html", "text/html"),
        "ieeexploreapi.ieee.org/api/v1/search/articles": (
            "http-venue-eda-ieee-conflict.json",
            "application/json",
        ),
        "dl.acm.org/doi/proceedings/10.1145/3649329": ("http-venue-eda-acm-conflict.html", "text/html"),
    })
    transport = ControlledHTTPTransport(
        "operator@example.test",
        opener=opener,
        runtime=_runtime(),
        environment={"IEEE_XPLORE_API_KEY": "ieee-secret"},
    )
    descriptor = VenueDescriptor(
        1,
        "dac",
        "eda_proceedings",
        "eda_proceedings",
        {
            "series": "DAC",
            "upstreams": ["ieee_xplore", "acm_dl"],
            "resolve_platforms_by_year": True,
            "deduplicate_by": "doi",
            "official_routes_by_year": _official_eda_routes("DAC"),
            "upstream_routes_by_year": {
                2024: {
                    "ieee_xplore": {"publication_number": 11293840, "required": False},
                    "acm_dl": {"proceedings_doi": "10.1145/3649329", "required": True},
                }
            },
        },
    )

    batch = create_builtin("eda_proceedings", transport).discover(descriptor, CrawlWindow(year=2024))

    conflicted = next(entry for entry in batch.entries if entry.external_id == "DAC-2024-RESEARCH1860")
    assert batch.status.value == "partial"
    assert "conflicting upstream DOIs" in (batch.error or "")
    assert conflicted.doi is None
    assert conflicted.title == "Scalable Chip Design"
    assert conflicted.abstract == "DAC abstract."
    assert conflicted.metadata["upstream_resolution"]["status"] == "conflicted"
    assert {item["doi"] for item in conflicted.metadata["upstream_resolution"]["doi_candidates"]} == {
        "10.1109/test.dac.2024.00001",
        "10.1145/3649329.test9999",
    }


def test_eda_requires_explicit_required_route_flags() -> None:
    transport, _ = _transport({})
    with pytest.raises(ValueError, match="explicit required boolean"):
        transport(
            "eda_proceedings",
            "discover",
            {
                "series": "DAC",
                "upstreams": ["ieee_xplore", "acm_dl"],
                "resolve_platforms_by_year": True,
                "deduplicate_by": "doi",
                "year": 2024,
                "official_routes_by_year": _official_eda_routes("DAC"),
                "upstream_routes_by_year": {
                    2024: {"ieee_xplore": {"publication_number": 11293840}}
                },
            },
        )


def test_eda_rejects_mismatched_official_and_upstream_ieee_publications() -> None:
    transport, _ = _transport({})
    with pytest.raises(ValueError, match="same publication_number"):
        transport(
            "eda_proceedings",
            "discover",
            {
                "series": "DAC",
                "upstreams": ["ieee_xplore", "acm_dl"],
                "resolve_platforms_by_year": True,
                "deduplicate_by": "doi",
                "year": 2024,
                "official_routes_by_year": {
                    2024: {
                        "route_kind": "ieee_xplore_publication",
                        "publication_number": 11293840,
                        "evidence_url": "https://ieeexplore.ieee.org/xpl/conhome/11293840/proceeding",
                    }
                },
                "upstream_routes_by_year": {
                    2024: {
                        "ieee_xplore": {"publication_number": 10247654, "required": True}
                    }
                },
            },
        )


def test_default_eda_manifest_denies_live_fetch_until_terms_are_accepted() -> None:
    transport = ControlledHTTPTransport(
        "operator@example.test", opener=lambda *_: pytest.fail("policy denial must happen before network")
    )
    with pytest.raises(ProviderPolicyDenied, match="terms are not accepted"):
        transport(
            "eda_proceedings",
            "discover",
            {
                "series": "ICCAD",
                "upstreams": ["ieee_xplore", "acm_dl"],
                "resolve_platforms_by_year": True,
                "deduplicate_by": "doi",
                "official_routes_by_year": _official_eda_routes("ICCAD"),
                "upstream_routes_by_year": {
                    2024: {
                        "ieee_xplore": {"publication_number": 11126043, "required": True}
                    }
                },
                "year": 2024,
            },
        )


def test_eda_never_guesses_an_unfrozen_year_route() -> None:
    transport, _ = _transport({})
    with pytest.raises(ProviderRequestError, match="no frozen official DAC route for 2023"):
        transport(
            "eda_proceedings",
            "discover",
            {
                "series": "DAC",
                "upstreams": ["ieee_xplore", "acm_dl"],
                "resolve_platforms_by_year": True,
                "deduplicate_by": "doi",
                "official_routes_by_year": _official_eda_routes("DAC"),
                "year": 2023,
            },
        )


def test_default_eda_runtime_uses_separate_frozen_terms_and_limits_for_each_upstream() -> None:
    opener = FixtureOpener({
        "61dac.conference-program.com/search-program/": ("http-venue-dac.html", "text/html"),
        "ieeexploreapi.ieee.org/api/v1/search/articles": ("http-venue-eda-ieee.json", "application/json"),
        "dl.acm.org/doi/proceedings/10.1145/3649329": ("http-venue-eda-acm.html", "text/html"),
    })
    transport = ControlledHTTPTransport(
        "operator@example.test",
        opener=opener,
        environment={"IEEE_XPLORE_API_KEY": "ieee-secret"},
        accepted_terms={
            "eda_proceedings:dac_program": "https://www.dac.com/",
            "eda_proceedings:ieee_xplore": "https://www.ieee.org/site-terms-conditions.html",
            "eda_proceedings:acm_dl": "https://www.acm.org/publications/policies/terms-of-use",
        },
    )

    payload = transport(
        "eda_proceedings",
        "discover",
        {
            "series": "DAC",
            "upstreams": ["ieee_xplore", "acm_dl"],
            "resolve_platforms_by_year": True,
            "deduplicate_by": "doi",
            "official_routes_by_year": _official_eda_routes("DAC"),
            "year": 2024,
            "upstream_routes_by_year": {
                2024: {
                    "ieee_xplore": {"publication_number": 11293840, "required": False},
                    "acm_dl": {"proceedings_doi": "10.1145/3649329", "required": True},
                }
            },
        },
    )

    assert payload["status"] == "success"
    assert [record["provider"] for record in payload["_request_audit"]] == [
        "eda_proceedings:dac_program",
        "eda_proceedings:ieee_xplore",
        "eda_proceedings:acm_dl",
    ]


@pytest.mark.parametrize("cursor", ["-1", "page-two"])
def test_local_page_providers_reject_non_offset_cursors(cursor: str) -> None:
    transport, _ = _transport({
        "proceedings.neurips.cc/paper_files/paper/2024": ("http-venue-neurips.html", "text/html"),
    })
    with pytest.raises(ValueError, match="non-negative integer"):
        transport("neurips_proceedings", "discover", {"series": "NeurIPS", "year": 2024, "cursor": cursor})


def test_openreview_surfaces_native_api_error_envelope() -> None:
    class ErrorOpener:
        def __call__(self, request, timeout):
            return Response(b'{"name":"ChallengeRequiredError"}', "application/json")

    transport = ControlledHTTPTransport("operator@example.test", opener=ErrorOpener(), runtime=_runtime())
    with pytest.raises(ProviderRequestError, match="ChallengeRequiredError"):
        transport(
            "openreview",
            "discover",
            {
                "invitation": "ICLR.cc/2024/Conference/-/Submission",
                "api_version": "v2",
                "year": 2024,
            },
        )


def test_controlled_transport_accepts_an_independent_provider_family_delegate() -> None:
    def execute(provider, operation, parameters, fetch):
        assert (provider, operation, parameters) == ("aaai_ojs", "discover", {"year": 2024})
        return VenueOperationResult({"entries": [{"id": "delegated", "title": "Delegated"}]})

    transport = ControlledHTTPTransport(
        "operator@example.test",
        opener=lambda *_: pytest.fail("delegate owns any network operation"),
        runtime=_runtime(),
        delegates=(HTTPProviderDelegate(("aaai_ojs",), execute),),
    )
    assert transport("aaai_ojs", "discover", {"year": 2024})["entries"][0]["id"] == "delegated"


@pytest.mark.parametrize(
    ("provider", "fixture", "environment_name", "parameters", "query_key", "header_name"),
    [
        (
            "ieee_xplore",
            "ieee-xplore-native.json",
            "IEEE_XPLORE_API_KEY",
            {"publication_number": 43, "issns": ["0278-0070"], "year": 2024},
            "apikey",
            None,
        ),
        (
            "springer_nature",
            "springer-native.json",
            "SPRINGER_NATURE_API_KEY",
            {"journal_slug": "natmachintell", "issns": ["2522-5839"], "article_types": ["Article"], "year": 2024},
            "api_key",
            None,
        ),
        (
            "cell_press",
            "cell-elsevier-native.json",
            "ELSEVIER_API_KEY",
            {"issn": "0092-8674", "year": 2024},
            None,
            "X-els-apikey",
        ),
    ],
)
def test_controlled_journal_delegate_injects_and_redacts_declared_credentials(
    provider: str,
    fixture: str,
    environment_name: str,
    parameters: dict[str, object],
    query_key: str | None,
    header_name: str | None,
) -> None:
    opener = FixtureOpener({"https://": (fixture, "application/json")})
    runtime = ProviderRuntime(
        {
            provider: ProviderRuntimePolicy(
                provider,
                credentials_required=True,
                credential_environment_variables=(environment_name,),
                retry_attempts=1,
                jitter_seconds=0,
            )
        }
    )
    secret = f"{provider}-secret"
    transport = ControlledHTTPTransport(
        "operator@example.test",
        opener=opener,
        runtime=runtime,
        environment={environment_name: secret},
    )

    payload = transport(provider, "discover", parameters)

    request = opener.calls[0][0]
    if query_key:
        assert parse_qs(urlsplit(request.full_url).query)[query_key] == [secret]
        assert parse_qs(urlsplit(transport.last_request_url or "").query)[query_key] == ["<redacted>"]
    if header_name:
        assert request.get_header(header_name) == secret
    assert secret not in (transport.last_request_url or "")
    assert payload["entries"]

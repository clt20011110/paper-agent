"""Official field hydration for the standalone Stage 1 membership census.

Hydrators never discover papers.  They receive an already reconciled primary
membership set and may only return those exact external IDs with additional
fields and field-level provenance.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from hashlib import sha256
from html import unescape
from io import BytesIO
import json
import re
import tarfile
from threading import Lock
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, quote, urlencode, urlsplit
from xml.etree import ElementTree

import yaml

from .domain import SourceEntry
from .provider_runtime import ProviderRequestError
from .providers.api import VenueDescriptor
from .stage1 import Stage1HydrationResult


_OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}
_ATOM = "{http://www.w3.org/2005/Atom}"


@dataclass(slots=True)
class OfficialStage1FieldHydrator:
    """Dispatch manifest-selected official enrichment without changing census."""

    transport: Any
    _aaai_records: dict[str, Mapping[str, Any]] | None = field(
        default_factory=dict, init=False, repr=False
    )
    _aaai_warnings: tuple[str, ...] = field(default=(), init=False, repr=False)
    _aaai_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def hydrate(
        self,
        descriptor: VenueDescriptor,
        year: int,
        entries: Sequence[SourceEntry],
    ) -> Stage1HydrationResult:
        name = str(descriptor.parameters.get("field_enrichment") or "")
        if name == "aaai_oai":
            return self._aaai(year, entries)
        if name == "iclr_official":
            return self._iclr(year, entries)
        if name == "ijcai_official":
            return self._ijcai(year, entries)
        if name == "pmlr_official_snapshot":
            return self._pmlr(entries)
        if name == "neurips_official_export":
            return self._neurips(year, entries)
        if name == "jmlr_official_rss":
            return self._jmlr(entries)
        if name == "acl_official_registry":
            return self._acl(year, entries)
        if name == "cvf_official_registry":
            return self._cvf(descriptor, year, entries)
        raise ValueError(f"unknown Stage 1 field enrichment: {name}")

    def _cvf(
        self,
        descriptor: VenueDescriptor,
        year: int,
        entries: Sequence[SourceEntry],
    ) -> Stage1HydrationResult:
        series = str(descriptor.parameters.get("series") or "").upper()
        if series not in {"CVPR", "ICCV"}:
            raise ValueError("CVF hydration requires series=CVPR or series=ICCV")
        hashes: list[str] = []
        abstracts: dict[str, tuple[Mapping[str, Any], ...]] = {}
        if (series == "CVPR" and year >= 2023) or (series == "ICCV" and year >= 2025):
            response = self.transport.fetch_metadata(
                "cvf_open_access",
                f"https://{series.casefold()}.thecvf.com/static/virtual/data/"
                f"{series.casefold()}-{year}-orals-posters.json",
                api_version="cvf-official-virtual-papers-json-v1",
                request_key=f"{series}:{year}",
            )
            hashes.append(sha256(response.body).hexdigest())
            abstracts = _cvf_virtual_records(response.body)

        doi_response = self.transport.fetch_metadata(
            "dblp_toc",
            f"https://dblp.org/db/conf/{series.casefold()}/{series.casefold()}{year}.html",
            api_version="dblp-cvf-proceedings-html-v1",
            request_key=f"{series}:{year}",
        )
        hashes.append(sha256(doi_response.body).hexdigest())
        dois = _cvf_dblp_dois(doi_response.body)

        abstract_matches: dict[str, Mapping[str, Any]] = {}
        doi_matches: dict[str, str] = {}
        unresolved_abstracts: list[SourceEntry] = []
        unresolved_dois: list[SourceEntry] = []
        abstract_records = tuple(
            record for values in abstracts.values() for record in values
        )
        doi_records = tuple(record for values in dois.values() for record in values)
        for entry in entries:
            key = _normalize_title(entry.title)
            candidates = abstracts.get(key, ())
            if len(candidates) != 1 and abstract_records:
                fuzzy = _unique_fuzzy_title_match(entry, abstract_records)
                candidates = (fuzzy,) if fuzzy is not None else ()
            if len(candidates) == 1:
                abstract_matches[entry.external_id] = candidates[0]
            elif not entry.abstract:
                unresolved_abstracts.append(entry)

            doi_candidates = {str(record["doi"]) for record in dois.get(key, ())}
            if len(doi_candidates) != 1 and doi_records:
                fuzzy_doi = _unique_fuzzy_title_match(
                    entry, doi_records, require_abstract=False
                )
                doi_candidates = (
                    {str(fuzzy_doi["doi"])} if fuzzy_doi is not None else set()
                )
            if len(doi_candidates) == 1:
                doi_matches[entry.external_id] = next(iter(doi_candidates))
            elif not entry.doi:
                unresolved_dois.append(entry)

        if unresolved_abstracts:
            with ThreadPoolExecutor(max_workers=4) as executor:
                details = tuple(executor.map(self._cvf_detail_record, unresolved_abstracts))
            for entry, record, body_hash in details:
                abstract_matches[entry.external_id] = record
                hashes.append(body_hash)

        if unresolved_dois:
            with ThreadPoolExecutor(max_workers=4) as executor:
                registry = tuple(
                    executor.map(
                        lambda entry: self._cvf_crossref_record(series, year, entry),
                        unresolved_dois,
                    )
                )
            for entry, doi, body_hash in registry:
                if doi:
                    doi_matches[entry.external_id] = doi
                hashes.append(body_hash)

        hydrated: list[SourceEntry] = []
        missing_abstracts: list[str] = []
        for entry in entries:
            abstract_record = abstract_matches.get(entry.external_id)
            abstract = (
                _clean(abstract_record.get("abstract"))
                if abstract_record is not None
                else entry.abstract
            )
            pdf_url = (
                _clean(abstract_record.get("pdf_url"))
                if abstract_record is not None
                else None
            ) or entry.pdf_url
            doi = doi_matches.get(entry.external_id) or entry.doi
            if not abstract:
                missing_abstracts.append(entry.external_id)
            provenance = dict(entry.metadata.get("field_provenance") or {})
            if abstract:
                provenance["abstract"] = str(
                    abstract_record.get("_source")
                    if abstract_record is not None
                    else "cvf_open_access:primary"
                )
            if pdf_url:
                provenance["pdf_url"] = "cvf_open_access:public_pdf"
            overrides = dict(entry.metadata.get("field_status_overrides") or {})
            if doi:
                provenance["doi"] = (
                    "dblp_toc:registered_doi"
                    if entry.external_id in doi_matches
                    else "cvf_open_access:primary"
                )
            else:
                provenance["doi"] = "crossref_registry:not_registered"
                overrides["doi"] = "legitimately_absent"
            hydrated.append(
                replace(
                    entry,
                    abstract=abstract,
                    doi=doi,
                    pdf_url=pdf_url,
                    metadata={
                        **entry.metadata,
                        "field_status_overrides": overrides,
                        "doi_availability": None if doi else "not_registered",
                        "field_provenance": provenance,
                    },
                )
            )
        if missing_abstracts:
            raise ProviderRequestError(
                f"CVF official sources lack {len(missing_abstracts)} abstract(s): "
                f"{', '.join(missing_abstracts[:3])}"
            )
        return Stage1HydrationResult(tuple(hydrated), tuple(hashes), tuple(hashes))

    def _cvf_detail_record(
        self, entry: SourceEntry
    ) -> tuple[SourceEntry, Mapping[str, Any], str]:
        response = self.transport.fetch_metadata(
            "cvf_open_access",
            str(entry.landing_url),
            api_version="cvf-official-paper-detail-v1",
            request_key=entry.external_id,
        )
        return entry, _cvf_detail(response.body), sha256(response.body).hexdigest()

    def _cvf_crossref_record(
        self, series: str, year: int, entry: SourceEntry
    ) -> tuple[SourceEntry, str | None, str]:
        response = self.transport.fetch_metadata(
            "crossref",
            "https://api.crossref.org/works?"
            + urlencode(
                {
                    "filter": (
                        f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31,"
                        "prefix:10.1109,type:proceedings-article"
                    ),
                    "query.title": entry.title,
                    "rows": 20,
                    "select": "DOI,title,container-title",
                }
            ),
            api_version="crossref-cvf-title-audit-v1",
            request_key=entry.external_id,
        )
        return (
            entry,
            _cvf_crossref_exact_doi(response.body, series, entry.title),
            sha256(response.body).hexdigest(),
        )

    def _acl(
        self, year: int, entries: Sequence[SourceEntry]
    ) -> Stage1HydrationResult:
        registry: dict[str, Mapping[str, Any]] = {}
        hashes: list[str] = []
        cursor = "*"
        page_number = 0
        while cursor:
            response = self.transport.fetch_metadata(
                "crossref",
                "https://api.crossref.org/prefixes/10.18653/works?"
                + urlencode(
                    {
                        "filter": (
                            f"from-pub-date:{year}-01-01,"
                            f"until-pub-date:{year}-12-31"
                        ),
                        "rows": 1000,
                        "cursor": cursor,
                        "select": "DOI,title,abstract,container-title",
                    }
                ),
                api_version="crossref-acl-year-enrichment-v1",
                request_key=f"{year}:{page_number}",
            )
            hashes.append(sha256(response.body).hexdigest())
            page, cursor = _acl_crossref_page(response.body)
            registry.update(page)
            page_number += 1

        hydrated: list[SourceEntry] = []
        for entry in entries:
            record = registry.get(entry.external_id.casefold())
            abstract = entry.abstract or (
                _clean(record.get("abstract")) if record is not None else None
            )
            doi = entry.doi or (_clean(record.get("doi")) if record is not None else None)
            abstract_source = "acl_anthology:pinned_xml.abstract"
            if not abstract:
                response = self.transport.fetch_metadata(
                    "acl_anthology",
                    str(entry.pdf_url),
                    api_version="acl-official-pdf-abstract-fallback-v1",
                    request_key=entry.external_id,
                )
                hashes.append(sha256(response.body).hexdigest())
                abstract = _acl_first_page_abstract(response.body)
                abstract_source = "acl_anthology:official_pdf.first_page_abstract"
            if not abstract:
                raise ProviderRequestError(
                    f"ACL paper {entry.external_id} has no extractable official abstract"
                )
            provenance = dict(entry.metadata.get("field_provenance") or {})
            provenance["abstract"] = abstract_source
            provenance["pdf_url"] = "acl_anthology:public_pdf"
            overrides = dict(entry.metadata.get("field_status_overrides") or {})
            if doi:
                provenance["doi"] = (
                    "acl_anthology:pinned_xml.doi"
                    if entry.doi
                    else "crossref_registry:message.DOI"
                )
            else:
                provenance["doi"] = "acl_crossref_registry:not_registered"
                overrides["doi"] = "legitimately_absent"
            hydrated.append(
                replace(
                    entry,
                    abstract=abstract,
                    doi=doi,
                    metadata={
                        **entry.metadata,
                        "field_status_overrides": overrides,
                        "doi_availability": (
                            None if doi else "not_registered_by_proceedings"
                        ),
                        "field_provenance": provenance,
                    },
                )
            )
        return Stage1HydrationResult(tuple(hydrated), tuple(hashes), tuple(hashes))

    def _jmlr(self, entries: Sequence[SourceEntry]) -> Stage1HydrationResult:
        response = self.transport.fetch_metadata(
            "jmlr_official",
            "https://www.jmlr.org/jmlr.xml",
            api_version="jmlr-official-rss-v1",
            request_key="all",
        )
        records = _jmlr_rss_records(response.body)
        hashes = [sha256(response.body).hexdigest()]
        unresolved = tuple(entry for entry in entries if entry.external_id not in records)
        if unresolved:
            with ThreadPoolExecutor(max_workers=4) as executor:
                fetched = tuple(executor.map(self._jmlr_detail_record, unresolved))
            for entry, record, body_hash in fetched:
                records[entry.external_id] = record
                hashes.append(body_hash)
        hydrated: list[SourceEntry] = []
        missing: list[str] = []
        for entry in entries:
            record = records.get(entry.external_id)
            abstract = _clean(record.get("abstract")) if record else None
            if not abstract:
                missing.append(entry.external_id)
            provenance = dict(entry.metadata.get("field_provenance") or {})
            if abstract:
                provenance["abstract"] = "jmlr_rss:item.description"
            provenance["doi"] = "jmlr_bibliography:not_assigned_by_journal"
            provenance["pdf_url"] = "jmlr_rss:item.pdf"
            overrides = dict(entry.metadata.get("field_status_overrides") or {})
            overrides["doi"] = "legitimately_absent"
            hydrated.append(
                replace(
                    entry,
                    abstract=abstract,
                    pdf_url=_clean(record.get("pdf_url")) if record else entry.pdf_url,
                    metadata={
                        **entry.metadata,
                        "field_status_overrides": overrides,
                        "doi_availability": "not_assigned_by_journal",
                        "field_provenance": provenance,
                    },
                )
            )
        if missing:
            raise ProviderRequestError(
                f"JMLR official RSS lacks {len(missing)} census abstract(s): "
                f"{', '.join(missing[:3])}"
            )
        return Stage1HydrationResult(tuple(hydrated), tuple(hashes), tuple(hashes))

    def _jmlr_detail_record(
        self, entry: SourceEntry
    ) -> tuple[SourceEntry, Mapping[str, Any], str]:
        response = self.transport.fetch_metadata(
            "jmlr_official",
            str(entry.landing_url),
            api_version="jmlr-official-paper-detail-v1",
            request_key=entry.external_id,
        )
        return entry, _jmlr_detail(response.body), sha256(response.body).hexdigest()

    def _neurips(
        self, year: int, entries: Sequence[SourceEntry]
    ) -> Stage1HydrationResult:
        if year >= 2020:
            export = self.transport.fetch_metadata(
                "neurips_proceedings",
                f"https://neurips.cc/static/virtual/data/neurips-{year}-orals-posters.json",
                api_version="neurips-official-static-poster-json-v1",
                request_key=str(year),
            )
            form = None
        else:
            form, export = self.transport.fetch_form_export(
                "neurips_proceedings",
                f"https://neurips.cc/Downloads/{year}",
                form_values={
                    "format": "5",
                    "posters": "on",
                    "resource": "0",
                    "submitaction": "Download Data",
                },
                api_version="neurips-official-poster-json-v1",
                request_key=str(year),
            )
        by_title = _neurips_export_records(export.body)
        hashes = (
            [sha256(form.body).hexdigest(), sha256(export.body).hexdigest()]
            if form is not None
            else [sha256(export.body).hexdigest()]
        )
        dois: dict[str, str] = {}
        if year >= 2022:
            dois, registry_hashes = self._neurips_doi_registry(year)
            hashes.extend(registry_hashes)
        hydrated: list[SourceEntry] = []
        missing: list[str] = []
        ambiguous: list[str] = []
        for entry in entries:
            candidates = by_title.get(_normalize_title(entry.title), ())
            if not candidates:
                fuzzy = _unique_fuzzy_title_match(
                    entry,
                    tuple(record for values in by_title.values() for record in values),
                )
                candidates = (fuzzy,) if fuzzy is not None else ()
            abstracts = {str(record["abstract"]) for record in candidates}
            need_detail_abstract = len(abstracts) != 1
            doi = dois.get(_normalize_title(entry.title)) or entry.doi
            need_detail_doi = year >= 2022 and not doi
            detail_fields: Mapping[str, Any] = {}
            if need_detail_abstract or need_detail_doi:
                response = self.transport.fetch_metadata(
                    "neurips_proceedings",
                    str(entry.landing_url),
                    api_version="neurips-paper-detail-abstract-v1",
                    request_key=entry.external_id,
                )
                hashes.append(sha256(response.body).hexdigest())
                detail_fields = _neurips_detail(response.body)
            if need_detail_abstract:
                abstract = _clean(detail_fields.get("abstract")) or entry.abstract
                if not abstract:
                    (ambiguous if candidates else missing).append(entry.external_id)
            else:
                abstract = next(iter(abstracts))
            doi = doi or _clean(detail_fields.get("doi"))
            provenance = dict(entry.metadata.get("field_provenance") or {})
            if abstract:
                provenance["abstract"] = "neurips_official_export:poster.abstract"
            if doi:
                provenance["doi"] = (
                    "neurips_proceedings:citation_doi"
                    if detail_fields.get("doi")
                    else "crossref_registry:message.DOI"
                )
            elif year <= 2021:
                provenance["doi"] = (
                    "neurips_2016_2021_proceedings:not_assigned_by_venue"
                )
            provenance["pdf_url"] = "neurips_proceedings:public_pdf"
            overrides = dict(entry.metadata.get("field_status_overrides") or {})
            if year <= 2021:
                overrides["doi"] = "legitimately_absent"
            hydrated.append(
                replace(
                    entry,
                    abstract=abstract,
                    doi=doi,
                    metadata={
                        **entry.metadata,
                        "field_status_overrides": overrides,
                        "doi_availability": (
                            "not_assigned_by_venue" if year <= 2021 else None
                        ),
                        "field_provenance": provenance,
                    },
                )
            )
        if missing or ambiguous:
            raise ProviderRequestError(
                f"NeurIPS official export title join has {len(missing)} missing and "
                f"{len(ambiguous)} ambiguous census record(s)"
            )
        return Stage1HydrationResult(
            tuple(hydrated),
            tuple(hashes),
            tuple(hashes),
        )

    def _neurips_doi_registry(
        self, year: int
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        container = f"Advances in Neural Information Processing Systems {year - 1987}"
        output: dict[str, str] = {}
        hashes: list[str] = []
        cursor = "*"
        page_number = 0
        while cursor:
            response = self.transport.fetch_metadata(
                "crossref",
                "https://api.crossref.org/works?"
                + urlencode(
                    {
                        "filter": (
                            f"from-pub-date:{year}-01-01,"
                            f"until-pub-date:{year}-12-31,prefix:10.52202"
                        ),
                        "rows": 1000,
                        "cursor": cursor,
                        "select": "DOI,title,container-title",
                    }
                ),
                api_version="crossref-neurips-year-enrichment-v1",
                # Crossref deep cursors can remain byte-identical while their
                # server-side session advances. Page number prevents replaying
                # the first cached response forever.
                request_key=(
                    f"{year}:{page_number}:"
                    f"{sha256(cursor.encode()).hexdigest()[:12]}"
                ),
            )
            hashes.append(sha256(response.body).hexdigest())
            page, cursor = _neurips_crossref_page(response.body, container)
            page_number += 1
            for title, doi in page.items():
                prior = output.get(title)
                if prior is not None and prior != doi:
                    raise ProviderRequestError(
                        f"NeurIPS registry title maps to conflicting DOI: {title}"
                    )
                output[title] = doi
        return output, tuple(hashes)

    def _pmlr(self, entries: Sequence[SourceEntry]) -> Stage1HydrationResult:
        volumes = {
            entry.external_id.split("/", 1)[0]
            for entry in entries
            if entry.external_id.startswith("v") and "/" in entry.external_id
        }
        records: dict[str, Mapping[str, Any]] = {}
        hashes: list[str] = []
        for volume in sorted(volumes):
            response = self.transport.fetch_metadata(
                "pmlr",
                f"https://codeload.github.com/mlresearch/{volume}/tar.gz/refs/heads/gh-pages",
                api_version="pmlr-official-github-frontmatter-v1",
                request_key=volume,
            )
            hashes.append(sha256(response.body).hexdigest())
            records.update(_pmlr_frontmatter_snapshot(response.body, volume))

        hydrated: list[SourceEntry] = []
        missing: list[str] = []
        for entry in entries:
            record = records.get(entry.external_id)
            abstract = _clean(record.get("abstract")) if record else entry.abstract
            pdf_url = _clean(record.get("pdf")) if record else entry.pdf_url
            if not abstract or not pdf_url:
                missing.append(entry.external_id)
            provenance = dict(entry.metadata.get("field_provenance") or {})
            if abstract:
                provenance["abstract"] = (
                    "pmlr_official_github:frontmatter.abstract"
                    if record
                    else "pmlr_primary:legacy_abstract"
                )
            provenance["doi"] = "pmlr_frontmatter:not_assigned_by_venue"
            if pdf_url:
                provenance["pdf_url"] = (
                    "pmlr_official_github:frontmatter.pdf"
                    if record
                    else "pmlr_primary:official_pdf"
                )
            overrides = dict(entry.metadata.get("field_status_overrides") or {})
            overrides["doi"] = "legitimately_absent"
            hydrated.append(
                replace(
                    entry,
                    abstract=abstract,
                    pdf_url=pdf_url,
                    metadata={
                        **entry.metadata,
                        "field_status_overrides": overrides,
                        "doi_availability": "not_assigned_by_venue",
                        "field_provenance": provenance,
                    },
                )
            )
        if missing:
            raise ProviderRequestError(
                f"PMLR official snapshot lacks abstract/PDF for {len(missing)} "
                f"census record(s): {', '.join(missing[:3])}"
            )
        return Stage1HydrationResult(tuple(hydrated), tuple(hashes), tuple(hashes))

    def _ijcai(
        self, year: int, entries: Sequence[SourceEntry]
    ) -> Stage1HydrationResult:
        records: dict[str, Mapping[str, Any]] = {}
        hashes: list[str] = []
        cursor = "*" if year >= 2017 else None
        while cursor:
            response = self.transport.fetch_metadata(
                "crossref",
                "https://api.crossref.org/prefixes/10.24963/works?"
                + urlencode(
                    {
                        "filter": (
                            f"from-pub-date:{year}-01-01,"
                            f"until-pub-date:{year}-12-31"
                        ),
                        "rows": 1000,
                        "cursor": cursor,
                        "select": "DOI,title,abstract,link,published",
                    }
                ),
                api_version="crossref-ijcai-year-enrichment-v1",
                request_key=f"{year}:{sha256(cursor.encode()).hexdigest()[:12]}",
            )
            hashes.append(sha256(response.body).hexdigest())
            page, cursor = _ijcai_crossref_page(response.body, year)
            records.update(page)
        missing_entries = tuple(
            entry
            for entry in entries
            if entry.external_id.rsplit("-", 1)[-1] not in records
        )
        if year == 2016 and missing_entries:
            # The legacy proceedings expose one official abstract page per
            # paper. Fetch a small, policy-limited parallel window so the
            # decade census remains practical without trusting a third-party
            # title match or manufacturing an unregistered DOI.
            with ThreadPoolExecutor(max_workers=4) as executor:
                fetched = tuple(
                    executor.map(self._ijcai_detail_record, missing_entries)
                )
            for entry, fields, body_hash in fetched:
                records[entry.external_id.rsplit("-", 1)[-1]] = fields
                hashes.append(body_hash)
        hydrated: list[SourceEntry] = []
        for entry in entries:
            paper_id = entry.external_id.rsplit("-", 1)[-1]
            fields = records.get(paper_id)
            if fields is None:
                _, fields, body_hash = self._ijcai_detail_record(entry)
                hashes.append(body_hash)
            abstract = _clean(fields.get("abstract"))
            doi = _clean(fields.get("doi"))
            pdf_url = _clean(fields.get("pdf_url")) or entry.pdf_url
            provenance = dict(entry.metadata.get("field_provenance") or {})
            overrides = dict(entry.metadata.get("field_status_overrides") or {})
            source = str(fields.get("_source") or "ijcai_official")
            if abstract:
                provenance["abstract"] = f"{source}:abstract"
            if doi:
                provenance["doi"] = f"{source}:doi"
            elif year == 2016:
                provenance["doi"] = "ijcai_2016_proceedings:not_assigned_by_venue"
                overrides["doi"] = "legitimately_absent"
            if pdf_url:
                provenance["pdf_url"] = f"{source}:pdf"
            hydrated.append(
                replace(
                    entry,
                    abstract=abstract or entry.abstract,
                    doi=doi or entry.doi,
                    pdf_url=pdf_url,
                    metadata={
                        **entry.metadata,
                        "field_status_overrides": overrides,
                        "doi_availability": (
                            "not_assigned_by_venue" if year == 2016 else None
                        ),
                        "field_provenance": provenance,
                    },
                )
            )
        return Stage1HydrationResult(
            tuple(hydrated), tuple(hashes), tuple(hashes)
        )

    def _ijcai_detail_record(
        self, entry: SourceEntry
    ) -> tuple[SourceEntry, Mapping[str, Any], str]:
        response = self.transport.fetch_metadata(
            "ijcai_proceedings",
            str(entry.landing_url),
            api_version="ijcai-paper-detail-html-v1",
            request_key=entry.external_id,
        )
        return entry, _ijcai_detail(response.body), sha256(response.body).hexdigest()

    def _aaai(
        self, year: int, entries: Sequence[SourceEntry]
    ) -> Stage1HydrationResult:
        records, oai_hashes = self._aaai_records_for(year, entries)
        hydrated: list[SourceEntry] = []
        missing: list[str] = []
        extra_hashes: list[str] = []
        for entry in entries:
            record = records.get(entry.external_id)
            if record is None:
                missing.append(entry.external_id)
                hydrated.append(entry)
                continue
            abstract = _clean(record.get("abstract"))
            doi = _clean(record.get("doi"))
            pdf_url = _clean(record.get("pdf_url")) or entry.pdf_url
            field_source = str(record.get("_source") or "aaai_oai")
            abstract_source = (
                "crossref_registry:message.abstract"
                if field_source == "crossref"
                else "aaai_oai:oai_dc:dc.description"
            )
            if _abstract_suspicious(abstract) and pdf_url:
                pdf_abstract, body_hash = self._aaai_pdf_abstract(
                    entry.external_id, pdf_url
                )
                abstract = pdf_abstract
                extra_hashes.append(body_hash)
                abstract_source = "aaai_official_pdf:first_page_abstract"
            provenance = dict(entry.metadata.get("field_provenance") or {})
            if abstract:
                provenance["abstract"] = abstract_source
            if doi:
                provenance["doi"] = (
                    "crossref_registry:message.DOI"
                    if field_source == "crossref"
                    else "aaai_oai:oai_dc:dc.identifier"
                )
            if pdf_url:
                provenance["pdf_url"] = (
                    "crossref_registry:message.link.application/pdf"
                    if field_source == "crossref" and record.get("pdf_url")
                    else "aaai_oai:oai_dc:dc.relation"
                    if field_source == "aaai_oai" and record.get("pdf_url")
                    else "aaai_ojs:issue_html:pdf_galley"
                )
            hydrated.append(
                replace(
                    entry,
                    abstract=abstract or entry.abstract,
                    doi=doi or entry.doi,
                    pdf_url=pdf_url,
                    metadata={
                        **entry.metadata,
                        "field_provenance": provenance,
                    },
                )
            )
        if missing:
            raise ProviderRequestError(
                f"AAAI OAI snapshot is missing {len(missing)} census article ID(s): "
                f"{', '.join(missing[:3])}"
            )
        return Stage1HydrationResult(
            tuple(hydrated),
            (*oai_hashes, *extra_hashes),
            (*oai_hashes, *extra_hashes),
            self._aaai_warnings,
        )

    def _aaai_records_for(
        self, year: int, entries: Sequence[SourceEntry]
    ) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
        hashes: list[str] = []
        with self._aaai_lock:
            records = self._aaai_records
            required = {entry.external_id for entry in entries}
            if required - set(records):
                cursor = "*"
                while cursor:
                    response = self.transport.fetch_metadata(
                        "crossref",
                        "https://api.crossref.org/journals/2374-3468/works?"
                        + urlencode(
                            {
                                "filter": (
                                    f"from-pub-date:{year}-01-01,"
                                    f"until-pub-date:{year}-12-31"
                                ),
                                "rows": 1000,
                                "cursor": cursor,
                                "select": "DOI,title,abstract,link,published",
                            }
                        ),
                        api_version="crossref-aaai-year-enrichment-v1",
                        request_key=f"{year}:{sha256(cursor.encode()).hexdigest()[:12]}",
                    )
                    hashes.append(sha256(response.body).hexdigest())
                    page, cursor = _aaai_crossref_page(response.body)
                    for article_id, record in page.items():
                        records[article_id] = _prefer_complete_record(
                            records.get(article_id), record
                        )
            missing = sorted(required - set(records))
            for article_id in missing:
                response = self.transport.fetch_metadata(
                    "aaai_ojs",
                    "https://ojs.aaai.org/index.php/AAAI/oai?"
                    + urlencode(
                        {
                            "verb": "GetRecord",
                            "metadataPrefix": "oai_dc",
                            "identifier": f"oai:ojs.aaai.org:article/{article_id}",
                        }
                    ),
                    api_version="aaai-oai-pmh-getrecord-oai-dc-v1",
                    request_key=article_id,
                )
                digest = sha256(response.body).hexdigest()
                hashes.append(digest)
                page, _ = _aaai_oai_page(response.body)
                record = page.get(article_id)
                if record is None:
                    raise ProviderRequestError(
                        f"AAAI OAI GetRecord omitted census article/{article_id}"
                    )
                records[article_id] = record
        return records, tuple(hashes)

    def _aaai_pdf_abstract(self, article_id: str, pdf_url: str) -> tuple[str, str]:
        response = self.transport.fetch_metadata(
            "aaai_ojs",
            pdf_url,
            api_version="aaai-official-pdf-abstract-fallback-v1",
            request_key=f"abstract:{article_id}",
        )
        if not response.body.startswith(b"%PDF-"):
            raise ProviderRequestError(
                f"AAAI article/{article_id} abstract fallback is not a PDF"
            )
        abstract = _aaai_first_page_abstract(response.body)
        if not abstract:
            raise ProviderRequestError(
                f"AAAI article/{article_id} official PDF has no extractable abstract"
            )
        return abstract, sha256(response.body).hexdigest()

    def _iclr(
        self, year: int, entries: Sequence[SourceEntry]
    ) -> Stage1HydrationResult:
        if year == 2016:
            return self._iclr_2016(entries)
        if 2017 <= year <= 2025:
            return self._iclr_openreview(year, entries)
        raise ProviderRequestError(f"ICLR official enrichment has no route for {year}")

    def _iclr_2016(self, entries: Sequence[SourceEntry]) -> Stage1HydrationResult:
        by_arxiv: dict[str, SourceEntry] = {}
        for entry in entries:
            arxiv_id = _arxiv_id(entry)
            if not arxiv_id:
                raise ProviderRequestError(
                    f"ICLR 2016 DBLP record has no arXiv edition: {entry.external_id}"
                )
            by_arxiv[arxiv_id] = entry
        summaries: dict[str, str] = {}
        hashes: list[str] = []
        for chunk in _chunks(tuple(by_arxiv), 40):
            url = "https://export.arxiv.org/api/query?" + urlencode(
                {"id_list": ",".join(chunk), "max_results": len(chunk)}
            )
            response = self.transport.fetch_metadata(
                "arxiv",
                url,
                api_version="arxiv-atom-iclr-enrichment-v1",
                request_key=sha256("|".join(chunk).encode()).hexdigest(),
            )
            digest = sha256(response.body).hexdigest()
            hashes.append(digest)
            summaries.update(_arxiv_atom_summaries(response.body))
        hydrated = []
        for arxiv_id, entry in by_arxiv.items():
            abstract = summaries.get(arxiv_id)
            provenance = dict(entry.metadata.get("field_provenance") or {})
            if abstract:
                provenance["abstract"] = "arxiv_atom:summary"
            provenance["doi"] = "iclr_proceedings:not_assigned_by_venue"
            provenance["pdf_url"] = "arxiv:public_pdf"
            hydrated.append(
                replace(
                    entry,
                    abstract=abstract,
                    arxiv_id=arxiv_id,
                    pdf_url=f"https://arxiv.org/pdf/{quote(arxiv_id, safe='.')}",
                    metadata={
                        **entry.metadata,
                        "field_status_overrides": {"doi": "legitimately_absent"},
                        "doi_availability": "not_assigned_by_venue",
                        "field_provenance": provenance,
                    },
                )
            )
        return Stage1HydrationResult(
            tuple(hydrated), tuple(hashes), tuple(hashes)
        )

    def _iclr_openreview(
        self, year: int, entries: Sequence[SourceEntry]
    ) -> Stage1HydrationResult:
        by_forum: dict[str, SourceEntry] = {}
        for entry in entries:
            forum_id = _openreview_id(entry)
            if forum_id is None and year == 2023 and entry.external_id == "conf/iclr/AndersonCD23":
                forum_id = "zzqBoIFOQ1"
            if forum_id is None:
                raise ProviderRequestError(
                    f"ICLR {year} DBLP record has no OpenReview forum ID: {entry.external_id}"
                )
            if forum_id in by_forum:
                raise ProviderRequestError(f"duplicate ICLR OpenReview forum ID: {forum_id}")
            by_forum[forum_id] = entry

        abstracts: dict[str, str] = {}
        abstract_sources: dict[str, str] = {}
        hashes: list[str] = []
        openreview_error: Exception | None = None
        try:
            for url in _openreview_queries(year):
                offset = 0
                while True:
                    page_url = f"{url}&limit=1000&offset={offset}"
                    response = self.transport.fetch_metadata(
                        "openreview",
                        page_url,
                        api_version="openreview-notes-iclr-enrichment-v1",
                        request_key=f"{year}:{offset}:{sha256(url.encode()).hexdigest()[:12]}",
                    )
                    digest = sha256(response.body).hexdigest()
                    hashes.append(digest)
                    try:
                        payload = json.loads(response.body)
                    except json.JSONDecodeError as error:
                        raise ProviderRequestError("OpenReview enrichment is not JSON") from error
                    notes = payload.get("notes") if isinstance(payload, Mapping) else None
                    if not isinstance(notes, list):
                        raise ProviderRequestError("OpenReview enrichment has no notes list")
                    for note in notes:
                        if not isinstance(note, Mapping):
                            continue
                        forum_id = str(note.get("forum") or note.get("id") or "")
                        content = note.get("content")
                        if forum_id and isinstance(content, Mapping):
                            abstract = _openreview_value(content.get("abstract"))
                            if abstract:
                                abstracts[forum_id] = str(abstract).strip()
                                abstract_sources[forum_id] = (
                                    "openreview:note.content.abstract"
                                )
                    if len(notes) < 1000:
                        break
                    offset += len(notes)
        except Exception as error:
            openreview_error = error

        if set(by_forum) - set(abstracts) and year in {2018, 2019}:
            official, page_hashes = self._iclr_legacy_virtual_snapshot(year)
            hashes.extend(page_hashes)
            for forum_id, record in official.items():
                if forum_id in by_forum and record.get("abstract"):
                    abstracts.setdefault(forum_id, str(record["abstract"]))
                    abstract_sources.setdefault(
                        forum_id, "iclr_virtual:poster.abstract"
                    )

        if set(by_forum) - set(abstracts) and year in {2020, 2021, 2022, 2023, 2024, 2025}:
            by_id, by_title, body_hash = self._iclr_virtual_snapshot(year)
            hashes.append(body_hash)
            for forum_id, entry in by_forum.items():
                record = by_id.get(forum_id) or by_title.get(_normalize_title(entry.title))
                if record is None:
                    record = _unique_fuzzy_title_match(entry, tuple(by_title.values()))
                if record and record.get("abstract"):
                    abstracts.setdefault(forum_id, str(record["abstract"]))
                    abstract_sources.setdefault(
                        forum_id, "iclr_virtual:bulk_json.abstract"
                    )

        still_missing = sorted(set(by_forum) - set(abstracts))
        if still_missing:
            arxiv_abstracts, arxiv_hashes = self._iclr_arxiv_title_fallback(
                tuple((forum_id, by_forum[forum_id]) for forum_id in still_missing)
            )
            hashes.extend(arxiv_hashes)
            for forum_id, abstract in arxiv_abstracts.items():
                abstracts[forum_id] = abstract
                abstract_sources[forum_id] = "arxiv_atom:title_matched_summary"

        missing = sorted(set(by_forum) - set(abstracts))
        context = f"; OpenReview error={openreview_error}" if openreview_error else ""
        warning = (
            f"ICLR {year} enrichment lacks {len(missing)} abstract(s): "
            f"{', '.join(missing[:3])}{context}"
            if missing
            else None
        )
        hydrated = []
        for forum_id, entry in by_forum.items():
            provenance = dict(entry.metadata.get("field_provenance") or {})
            if forum_id in abstract_sources:
                provenance["abstract"] = abstract_sources[forum_id]
            provenance.update(
                {
                    "doi": "iclr_proceedings:not_assigned_by_venue",
                    "pdf_url": "openreview:pdf_endpoint",
                }
            )
            overrides = dict(entry.metadata.get("field_status_overrides") or {})
            overrides["doi"] = "legitimately_absent"
            if forum_id not in abstracts:
                overrides["abstract"] = "enrichment_failed"
            hydrated.append(
                replace(
                    entry,
                    abstract=abstracts.get(forum_id),
                    landing_url=f"https://openreview.net/forum?id={quote(forum_id, safe='')}",
                    pdf_url=f"https://openreview.net/pdf?id={quote(forum_id, safe='')}",
                    metadata={
                        **entry.metadata,
                        "openreview_forum_id": forum_id,
                        "field_status_overrides": overrides,
                        "doi_availability": "not_assigned_by_venue",
                        "field_provenance": provenance,
                    },
                )
            )
        return Stage1HydrationResult(
            tuple(hydrated),
            tuple(hashes),
            tuple(hashes),
            (warning,) if warning else (),
        )

    def _iclr_arxiv_title_fallback(
        self, entries: Sequence[tuple[str, SourceEntry]]
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        output: dict[str, str] = {}
        hashes: list[str] = []
        for chunk in _chunks(tuple(range(len(entries))), 10):
            query = " OR ".join(
                f'ti:"{entries[index][1].title.replace(chr(34), "")}"'
                for index in chunk
            )
            response = self.transport.fetch_metadata(
                "arxiv",
                "https://export.arxiv.org/api/query?"
                + urlencode(
                    {
                        "search_query": query,
                        "start": 0,
                        "max_results": max(10, len(chunk) * 3),
                    }
                ),
                api_version="arxiv-atom-iclr-title-fallback-v1",
                request_key=sha256(query.encode()).hexdigest(),
            )
            hashes.append(sha256(response.body).hexdigest())
            records = _arxiv_atom_records(response.body)
            by_title = {
                _normalize_title(record["title"]): record
                for record in records
                if record.get("title") and record.get("abstract")
            }
            for index in chunk:
                forum_id, entry = entries[index]
                record = by_title.get(_normalize_title(entry.title))
                if record is not None:
                    output[forum_id] = str(record["abstract"])
        return output, tuple(hashes)

    def _iclr_virtual_snapshot(
        self, year: int
    ) -> tuple[
        dict[str, Mapping[str, Any]],
        dict[str, Mapping[str, Any]],
        str,
    ]:
        url = f"https://iclr.cc/static/virtual/data/iclr-{year}-orals-posters.json"
        response = self.transport.fetch_metadata(
            "iclr_official",
            url,
            api_version="iclr-virtual-json-v1",
            request_key=str(year),
        )
        try:
            payload = json.loads(response.body)
        except json.JSONDecodeError as error:
            raise ProviderRequestError("ICLR virtual snapshot is not JSON") from error
        results = payload.get("results") if isinstance(payload, Mapping) else None
        if not isinstance(results, list):
            raise ProviderRequestError("ICLR virtual snapshot has no results list")
        by_id: dict[str, Mapping[str, Any]] = {}
        by_title: dict[str, Mapping[str, Any]] = {}
        for record in results:
            if not isinstance(record, Mapping) or not record.get("name"):
                continue
            key = _normalize_title(str(record["name"]))
            forum_id = _virtual_openreview_id(record)
            if forum_id:
                prior = by_id.get(forum_id)
                if prior is None or (
                    not prior.get("abstract") and record.get("abstract")
                ):
                    by_id[forum_id] = record
            prior_title = by_title.get(key)
            if prior_title is None or (
                not prior_title.get("abstract") and record.get("abstract")
            ):
                by_title[key] = record
            # A highlighted oral can duplicate the accepted-poster event with a
            # synthetic event ID and slightly edited abstract.  Forum-ID joins
            # remain authoritative; ambiguous title fallback is disabled.
            elif _virtual_openreview_id(prior_title) != forum_id:
                by_title.pop(key, None)
        return by_id, by_title, sha256(response.body).hexdigest()

    def _iclr_legacy_virtual_snapshot(
        self, year: int
    ) -> tuple[dict[str, Mapping[str, Any]], tuple[str, ...]]:
        index_url = f"https://iclr.cc/virtual/{year}/papers.html?filter=titles"
        index = self.transport.fetch_metadata(
            "iclr_official",
            index_url,
            api_version="iclr-legacy-virtual-index-v1",
            request_key=str(year),
        )
        poster_paths = tuple(
            dict.fromkeys(
                match.group(1)
                for match in re.finditer(
                    rf'href=["\'](/virtual/{year}/poster/\d+)["\']',
                    index.body.decode("utf-8", errors="replace"),
                    re.I,
                )
            )
        )
        if not poster_paths:
            raise ProviderRequestError(f"ICLR {year} virtual index has no poster links")
        output: dict[str, Mapping[str, Any]] = {}
        hashes = [sha256(index.body).hexdigest()]
        for path in poster_paths:
            response = self.transport.fetch_metadata(
                "iclr_official",
                "https://iclr.cc" + path,
                api_version="iclr-legacy-virtual-poster-v1",
                request_key=path.rsplit("/", 1)[-1],
            )
            hashes.append(sha256(response.body).hexdigest())
            record = _legacy_virtual_poster(response.body)
            if record is not None:
                output[str(record["forum_id"])] = record
        return output, tuple(hashes)


def _pmlr_frontmatter_snapshot(
    body: bytes, volume: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    try:
        # The controlled transport transparently removes HTTP gzip framing;
        # fixture/replay bytes may still retain it, so auto-detect both forms.
        archive = tarfile.open(fileobj=BytesIO(body), mode="r:*")
    except tarfile.TarError as error:
        raise ProviderRequestError("PMLR official snapshot is not a gzip tar archive") from error
    with archive:
        for member in archive.getmembers():
            if not re.search(r"/_posts/[^/]+\.md$", member.name) or not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            try:
                text = extracted.read().decode("utf-8")
            except UnicodeDecodeError as error:
                raise ProviderRequestError(
                    f"PMLR official frontmatter is not UTF-8: {member.name}"
                ) from error
            match = re.match(r"---\s*\n(.*?)\n---(?:\s*\n|$)", text, re.S)
            if match is None:
                continue
            try:
                record = yaml.safe_load(match.group(1))
            except yaml.YAMLError as error:
                raise ProviderRequestError(
                    f"PMLR official frontmatter is malformed: {member.name}"
                ) from error
            if not isinstance(record, Mapping):
                continue
            identifier = _clean(record.get("id"))
            if identifier:
                output[f"{volume}/{identifier}"] = record
    if not output:
        raise ProviderRequestError("PMLR official snapshot contained no frontmatter records")
    return output


def _jmlr_rss_records(body: bytes) -> dict[str, Mapping[str, str | None]]:
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        raise ProviderRequestError("JMLR official RSS is malformed XML") from error
    output: dict[str, Mapping[str, str | None]] = {}
    for item in root.findall("./channel/item"):
        link = _clean(item.findtext("./link"))
        match = re.search(r"/papers/(v\d+)/([^/]+)\.html$", link or "", re.I)
        if match is None:
            continue
        identifier = f"{match.group(1).casefold()}/{match.group(2)}"
        output[identifier] = {
            "abstract": _clean(item.findtext("./description")),
            "pdf_url": (_clean(item.findtext("./pdf")) or "").replace(
                "http://", "https://", 1
            ) or None,
        }
    if not output:
        raise ProviderRequestError("JMLR official RSS contains no papers")
    return output


def _jmlr_detail(body: bytes) -> Mapping[str, str | None]:
    text = body.decode("utf-8", errors="replace")
    abstract = re.search(
        r'<p\s+class=["\']abstract["\'][^>]*>(.*?)</p>', text, re.I | re.S
    )
    pdf = re.search(
        r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)',
        text,
        re.I,
    )
    return {
        "abstract": (
            _clean(unescape(re.sub(r"<[^>]+>", "", abstract.group(1))))
            if abstract is not None
            else None
        ),
        "pdf_url": (
            pdf.group(1).replace("http://", "https://", 1)
            if pdf is not None
            else None
        ),
    }


def _acl_crossref_page(
    body: bytes,
) -> tuple[dict[str, Mapping[str, str | None]], str | None]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderRequestError("ACL Crossref enrichment is not JSON") from error
    message = payload.get("message") if isinstance(payload, Mapping) else None
    items = message.get("items") if isinstance(message, Mapping) else None
    if not isinstance(items, list):
        raise ProviderRequestError("ACL Crossref enrichment has no items list")
    output: dict[str, Mapping[str, str | None]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        doi = _clean(item.get("DOI"))
        match = re.fullmatch(r"10\.18653/v1/(.+)", doi or "", re.I)
        if match is None:
            continue
        output[match.group(1).casefold()] = {
            "doi": doi.casefold() if doi else None,
            "abstract": _clean(
                unescape(re.sub(r"<[^>]+>", " ", str(item.get("abstract") or "")))
            ),
        }
    cursor = _clean(message.get("next-cursor")) if isinstance(message, Mapping) else None
    if len(items) < 1000:
        cursor = None
    return output, cursor


def _acl_first_page_abstract(body: bytes) -> str | None:
    if not body.startswith(b"%PDF-"):
        raise ProviderRequestError("ACL official abstract fallback is not a PDF")
    try:
        from pypdf import PdfReader

        text = PdfReader(BytesIO(body)).pages[0].extract_text() or ""
    except Exception as error:
        raise ProviderRequestError("ACL official PDF text extraction failed") from error
    lines = [line.strip() for line in text.splitlines()]
    start = next(
        (index + 1 for index, line in enumerate(lines) if line.casefold() == "abstract"),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.fullmatch(r"(?:\d+\s+)?Introduction", lines[index], re.I)
        ),
        None,
    )
    return _clean(" ".join(lines[start:end])) if end is not None else None


def _neurips_export_records(
    body: bytes,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderRequestError("NeurIPS official export is not JSON") from error
    if isinstance(payload, Mapping):
        payload = payload.get("results")
    if not isinstance(payload, list):
        raise ProviderRequestError("NeurIPS official export is not a record list")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in payload:
        if not isinstance(record, Mapping):
            continue
        event_type = record.get("type") or record.get("eventtype")
        if event_type not in {"Poster", "Oral"}:
            continue
        title = _clean(record.get("name"))
        abstract = _clean(record.get("abstract"))
        if title and abstract:
            grouped.setdefault(_normalize_title(title), []).append(
                {**record, "abstract": abstract}
            )
    if not grouped:
        raise ProviderRequestError("NeurIPS official export contains no poster abstracts")
    return {key: tuple(values) for key, values in grouped.items()}


def _neurips_detail(body: bytes) -> Mapping[str, str | None]:
    text = body.decode("utf-8", errors="replace")
    match = re.search(
        r'<p\s+class=["\']paper-abstract["\'][^>]*>(.*?)</p>\s*(?:</p>)?',
        text,
        re.I | re.S,
    )
    doi = re.search(
        r'<meta\s+name=["\']citation_doi["\']\s+content=["\']([^"\']+)',
        text,
        re.I,
    )
    return {
        "abstract": (
            _clean(unescape(re.sub(r"<[^>]+>", " ", match.group(1))))
            if match is not None
            else None
        ),
        "doi": doi.group(1).casefold() if doi is not None else None,
    }


def _neurips_detail_abstract(body: bytes) -> str | None:
    return _clean(_neurips_detail(body).get("abstract"))


def _neurips_crossref_page(
    body: bytes, container: str
) -> tuple[dict[str, str], str | None]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderRequestError("NeurIPS Crossref enrichment is not JSON") from error
    message = payload.get("message") if isinstance(payload, Mapping) else None
    items = message.get("items") if isinstance(message, Mapping) else None
    if not isinstance(items, list):
        raise ProviderRequestError("NeurIPS Crossref enrichment has no items list")
    output: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        containers = item.get("container-title")
        if not isinstance(containers, list) or container not in containers:
            continue
        title = item.get("title")
        title = title[0] if isinstance(title, list) and title else title
        doi = _clean(item.get("DOI"))
        if title and doi:
            output[_normalize_title(str(title))] = doi.casefold()
    cursor = _clean(message.get("next-cursor")) if isinstance(message, Mapping) else None
    if len(items) < 1000:
        cursor = None
    return output, cursor


def _aaai_oai_page(body: bytes) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        raise ProviderRequestError("AAAI OAI response is malformed XML") from error
    error_node = root.find("./oai:error", _OAI_NS)
    if error_node is not None:
        raise ProviderRequestError(f"AAAI OAI error: {_clean(error_node.text)}")
    output: dict[str, Mapping[str, Any]] = {}
    for record in root.findall(".//oai:record", _OAI_NS):
        identifier = record.findtext("./oai:header/oai:identifier", namespaces=_OAI_NS) or ""
        match = re.fullmatch(r"oai:ojs\.aaai\.org:article/(\d+)", identifier.strip())
        if not match:
            continue
        values = lambda name: [
            _clean(node.text)
            for node in record.findall(f".//dc:{name}", _OAI_NS)
            if _clean(node.text)
        ]
        identifiers = values("identifier")
        relations = values("relation")
        doi = next((value.casefold() for value in identifiers if value.startswith("10.")), None)
        pdf_url = next(
            (
                value
                for value in relations
                if re.search(r"/article/(?:view|download)/\d+/\d+", value)
            ),
            None,
        )
        descriptions = values("description")
        output[match.group(1)] = {
            "abstract": descriptions[0] if descriptions else None,
            "doi": doi,
            "pdf_url": pdf_url,
            "_source": "aaai_oai",
        }
    token = root.findtext(".//oai:resumptionToken", namespaces=_OAI_NS)
    return output, _clean(token)


def _aaai_crossref_page(
    body: bytes,
) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderRequestError("AAAI Crossref enrichment is not JSON") from error
    message = payload.get("message") if isinstance(payload, Mapping) else None
    items = message.get("items") if isinstance(message, Mapping) else None
    if not isinstance(items, list):
        raise ProviderRequestError("AAAI Crossref enrichment has no items list")
    output: dict[str, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        doi = _clean(item.get("DOI"))
        match = re.search(r"\.([0-9]+)$", doi or "")
        if not match:
            continue
        abstract = _clean(re.sub(r"<[^>]+>", " ", str(item.get("abstract") or "")))
        links = item.get("link") if isinstance(item.get("link"), list) else []
        pdf_url = next(
            (
                _clean(link.get("URL"))
                for link in links
                if isinstance(link, Mapping)
                and str(link.get("content-type") or "").casefold() == "application/pdf"
            ),
            None,
        )
        output[match.group(1)] = {
            "abstract": abstract,
            "doi": doi.casefold() if doi else None,
            "pdf_url": pdf_url,
            "_source": "crossref",
        }
    next_cursor = _clean(message.get("next-cursor")) if isinstance(message, Mapping) else None
    if len(items) < 1000:
        next_cursor = None
    return output, next_cursor


def _records_conflict(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return any(
        left.get(field) and right.get(field) and left[field] != right[field]
        for field in ("doi", "pdf_url")
    )


def _prefer_complete_record(
    prior: Mapping[str, Any] | None, current: Mapping[str, Any]
) -> Mapping[str, Any]:
    if prior is None:
        return current
    output = dict(prior)
    replaced = False
    for field in ("abstract", "doi", "pdf_url"):
        value = current.get(field)
        if value and (
            not output.get(field)
            or (field == "abstract" and len(str(value)) > len(str(output[field])))
        ):
            output[field] = value
            replaced = True
    if replaced:
        output["_source"] = current.get("_source", output.get("_source"))
    return output


def _abstract_suspicious(value: str | None) -> bool:
    if value is None:
        return True
    return value[:1].islower()


def _aaai_first_page_abstract(body: bytes) -> str | None:
    try:
        from pypdf import PdfReader

        first_page = PdfReader(BytesIO(body)).pages[0].extract_text() or ""
    except Exception as error:
        raise ProviderRequestError("AAAI official PDF text extraction failed") from error
    lines = [line.strip() for line in first_page.splitlines()]
    start = next(
        (index + 1 for index, line in enumerate(lines) if line.casefold() == "abstract"),
        None,
    )
    if start is None:
        start = next(
            (
                index + 1
                for index, line in enumerate(lines)
                if "@" in line
            ),
            None,
        )
    if start is None:
        return None
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if re.fullmatch(r"(?:\d+\s+)?Introduction", lines[index], re.I)
        ),
        None,
    )
    if end is None:
        return None
    return _clean(" ".join(lines[start:end]))


def _arxiv_atom_summaries(body: bytes) -> dict[str, str]:
    return {
        str(record["arxiv_id"]): str(record["abstract"])
        for record in _arxiv_atom_records(body)
    }


def _arxiv_atom_records(body: bytes) -> tuple[Mapping[str, str], ...]:
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        raise ProviderRequestError("arXiv Atom enrichment is malformed XML") from error
    output: list[Mapping[str, str]] = []
    for entry in root.findall(f"./{_ATOM}entry"):
        identifier = entry.findtext(f"./{_ATOM}id") or ""
        match = re.search(r"arxiv\.org/abs/([^?#]+)", identifier, re.I)
        summary = _clean(entry.findtext(f"./{_ATOM}summary"))
        title = _clean(entry.findtext(f"./{_ATOM}title"))
        if match and summary and title:
            output.append(
                {
                    "arxiv_id": re.sub(r"v\d+$", "", match.group(1)),
                    "title": title,
                    "abstract": summary,
                }
            )
    return tuple(output)


def _openreview_queries(year: int) -> tuple[str, ...]:
    if year == 2017:
        prefix = f"ICLR.cc/{year}/conference/-/"
        api = "https://api.openreview.net/notes?invitation="
        names = ("submission", "Submission")
    elif year <= 2023:
        prefix = f"ICLR.cc/{year}/Conference/-/"
        api = "https://api.openreview.net/notes?invitation="
        names = ("Blind_Submission", "Submission", "submission")
    else:
        return (
            "https://api2.openreview.net/notes?"
            + urlencode({"content.venueid": f"ICLR.cc/{year}/Conference"}),
        )
    return tuple(api + quote(prefix + name, safe="") for name in names)


def _openreview_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value", value.get("values"))
    return value


def _openreview_id(entry: SourceEntry) -> str | None:
    candidates = [entry.landing_url, *entry.metadata.get("electronic_editions", ())]
    for value in candidates:
        if not value:
            continue
        parts = urlsplit(str(value))
        if parts.netloc.casefold().endswith("openreview.net"):
            identifier = parse_qs(parts.query).get("id", [None])[0]
            if identifier:
                return str(identifier)
    return None


def _virtual_openreview_id(record: Mapping[str, Any]) -> str | None:
    candidates: list[Any] = [record.get("paper_url"), record.get("paper_pdf_url")]
    eventmedia = record.get("eventmedia")
    if isinstance(eventmedia, list):
        candidates.extend(
            value.get("uri")
            for value in eventmedia
            if isinstance(value, Mapping)
        )
    for value in candidates:
        if not value:
            continue
        parts = urlsplit(str(value))
        if parts.netloc.casefold().endswith("openreview.net"):
            identifier = parse_qs(parts.query).get("id", [None])[0]
            if identifier:
                return str(identifier)
    return None


def _legacy_virtual_poster(body: bytes) -> Mapping[str, Any] | None:
    text = body.decode("utf-8", errors="replace")
    forum = re.search(
        r'href=["\']https://openreview\.net/(?:forum|pdf)\?id=([^"\'&]+)',
        text,
        re.I,
    )
    abstract = re.search(
        r'<div\s+class=["\']abstract-text-inner["\'][^>]*>(.*?)</div>',
        text,
        re.I | re.S,
    )
    if forum is None or abstract is None:
        return None
    abstract_text = re.sub(r"<[^>]+>", " ", abstract.group(1))
    abstract_text = _clean(
        abstract_text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    )
    if not abstract_text:
        return None
    return {"forum_id": forum.group(1), "abstract": abstract_text}


def _ijcai_detail(body: bytes) -> Mapping[str, str | None]:
    text = body.decode("utf-8", errors="replace")
    doi_match = re.search(
        r'href=["\']https://doi\.org/(10\.24963/ijcai\.\d{4}/\d+)["\']',
        text,
        re.I,
    )
    pdf_match = re.search(
        r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)',
        text,
        re.I,
    )
    section = re.search(
        r'<div\s+class=["\']col-md-12["\']>\s*(.*?)\s*</div>',
        text,
        re.I | re.S,
    )
    if section is None and "/Proceedings/16/Papers/" in text:
        paragraphs = re.findall(r"<p(?:\s[^>]*)?>(.*?)</p>", text, re.I | re.S)
        section = next(
            (
                re.match(r"(?s)(.*)", paragraph)
                for paragraph in paragraphs[1:]
                if "/Proceedings/16/Papers/" not in paragraph
            ),
            None,
        )
    abstract = (
        _clean(unescape(re.sub(r"<[^>]+>", " ", section.group(1))))
        if section is not None
        else None
    )
    if pdf_match is None:
        legacy_pdf = re.search(
            r'href=["\'](/Proceedings/16/Papers/\d+\.pdf)["\']', text, re.I
        )
        pdf_url = (
            "https://www.ijcai.org" + legacy_pdf.group(1)
            if legacy_pdf is not None
            else None
        )
    else:
        pdf_url = pdf_match.group(1)
    return {
        "abstract": abstract,
        "doi": doi_match.group(1).casefold() if doi_match else None,
        "pdf_url": pdf_url,
        "_source": "ijcai_official:paper_detail",
    }


def _ijcai_crossref_page(
    body: bytes, year: int
) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderRequestError("IJCAI Crossref enrichment is not JSON") from error
    message = payload.get("message") if isinstance(payload, Mapping) else None
    items = message.get("items") if isinstance(message, Mapping) else None
    if not isinstance(items, list):
        raise ProviderRequestError("IJCAI Crossref enrichment has no items list")
    output: dict[str, Mapping[str, Any]] = {}
    pattern = re.compile(rf"10\.24963/ijcai\.{year}/(\d+)$", re.I)
    for item in items:
        if not isinstance(item, Mapping):
            continue
        doi = _clean(item.get("DOI"))
        match = pattern.fullmatch(doi or "")
        if match is None:
            continue
        abstract = _clean(re.sub(r"<[^>]+>", " ", str(item.get("abstract") or "")))
        links = item.get("link") if isinstance(item.get("link"), list) else []
        pdf_url = next(
            (
                _clean(link.get("URL"))
                for link in links
                if isinstance(link, Mapping)
                and str(link.get("content-type") or "").casefold() == "application/pdf"
            ),
            None,
        )
        output[match.group(1)] = {
            "abstract": abstract,
            "doi": doi.casefold() if doi else None,
            "pdf_url": pdf_url,
            "_source": "crossref_registry:ijcai",
        }
    cursor = _clean(message.get("next-cursor")) if isinstance(message, Mapping) else None
    if len(items) < 1000:
        cursor = None
    return output, cursor


def _unique_fuzzy_title_match(
    entry: SourceEntry,
    records: Sequence[Mapping[str, Any]],
    *,
    require_abstract: bool = True,
) -> Mapping[str, Any] | None:
    target = _normalize_title(entry.title)
    scored = list(
        (
            SequenceMatcher(None, target, _normalize_title(str(record.get("name") or ""))).ratio(),
            str(record.get("name") or ""),
            record,
        )
        for record in records
        if record.get("name") and (record.get("abstract") or not require_abstract)
    )
    scored.sort(key=lambda item: (item[0], item[1]))
    if not scored:
        return None
    best_score, _, best = scored[-1]
    second_score = scored[-2][0] if len(scored) > 1 else 0.0
    if best_score >= 0.92 and best_score - second_score >= 0.02:
        return best
    return None


def _cvf_virtual_records(
    body: bytes,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderRequestError("CVF virtual export is not JSON") from error
    records = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(records, list):
        raise ProviderRequestError("CVF virtual export has no results list")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        if record.get("eventtype") not in {"Poster", "Oral"}:
            continue
        title = _clean(record.get("name"))
        abstract = _clean(record.get("abstract"))
        if title and abstract:
            grouped.setdefault(_normalize_title(title), []).append(
                {
                    "name": title,
                    "abstract": abstract,
                    "_source": "cvf_conference_virtual:annual_json.abstract",
                }
            )
    if not grouped:
        raise ProviderRequestError("CVF virtual export contains no paper abstracts")
    return {key: tuple(values) for key, values in grouped.items()}


def _cvf_dblp_dois(
    body: bytes,
) -> dict[str, tuple[Mapping[str, str], ...]]:
    text = body.decode("utf-8", errors="replace")
    grouped: dict[str, list[Mapping[str, str]]] = {}
    blocks = re.findall(
        r'<li\s+class=["\']entry inproceedings["\'].*?'
        r'(?=<li\s+class=["\']entry inproceedings["\']|$)',
        text,
        re.I | re.S,
    )
    for block in blocks:
        title_match = re.search(
            r'<span\s+class=["\']title["\'][^>]*>(.*?)</span>',
            block,
            re.I | re.S,
        )
        doi_match = re.search(
            r'https://doi\.org/(10\.1109/[^"\'<\s]+)', block, re.I
        )
        if title_match is None or doi_match is None:
            continue
        title = _clean(unescape(re.sub(r"<[^>]+>", " ", title_match.group(1))))
        doi = doi_match.group(1).casefold().rstrip(".,;)")
        if title:
            grouped.setdefault(_normalize_title(title), []).append(
                {"name": title, "doi": doi}
            )
    if not grouped:
        raise ProviderRequestError("DBLP CVF proceedings page contains no article DOI")
    return {key: tuple(values) for key, values in grouped.items()}


def _cvf_crossref_exact_doi(body: bytes, series: str, title: str) -> str | None:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ProviderRequestError("CVF Crossref title audit is not JSON") from error
    message = payload.get("message") if isinstance(payload, Mapping) else None
    items = message.get("items") if isinstance(message, Mapping) else None
    if not isinstance(items, list):
        raise ProviderRequestError("CVF Crossref title audit has no items list")
    target = _normalize_title(title)
    matches: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        titles = item.get("title")
        candidate = titles[0] if isinstance(titles, list) and titles else titles
        containers = item.get("container-title")
        container_text = " ".join(
            str(value) for value in containers
        ) if isinstance(containers, list) else str(containers or "")
        doi = _clean(item.get("DOI"))
        if (
            candidate
            and _normalize_title(str(candidate)) == target
            and re.search(rf"\b{re.escape(series)}\b", container_text, re.I)
            and doi
        ):
            matches.add(doi.casefold())
    return next(iter(matches)) if len(matches) == 1 else None


def _cvf_detail(body: bytes) -> Mapping[str, str | None]:
    text = body.decode("utf-8", errors="replace")
    abstract = re.search(
        r'<div\s+id=["\']abstract["\'][^>]*>(.*?)</div>', text, re.I | re.S
    )
    pdf = re.search(
        r'<meta\s+name=["\']citation_pdf_url["\']\s+content=["\']([^"\']+)',
        text,
        re.I,
    )
    return {
        "abstract": (
            _clean(
                re.sub(
                    r"\s+([.,;:!?])",
                    r"\1",
                    unescape(re.sub(r"<[^>]+>", " ", abstract.group(1))),
                )
            )
            if abstract is not None
            else None
        ),
        "pdf_url": _clean(unescape(pdf.group(1))) if pdf is not None else None,
        "_source": "cvf_open_access:paper_detail.abstract",
    }


def _arxiv_id(entry: SourceEntry) -> str | None:
    candidates = [entry.arxiv_id, entry.landing_url, *entry.metadata.get("electronic_editions", ())]
    for value in candidates:
        match = re.search(r"(?:arxiv\.org/abs/|arxiv:)([^?#]+)", str(value or ""), re.I)
        if match:
            return re.sub(r"v\d+$", "", match.group(1)).rstrip("/")
    return None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def _normalize_title(value: str) -> str:
    value = unescape(value)
    value = re.sub(r"\\[A-Za-z]+", "", value)
    value = value.casefold().replace("ℓ", "l").replace("∞", "infinity")
    return "".join(character for character in value if character.isalnum())


def _chunks(values: Sequence[Any], size: int) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(values[index : index + size]) for index in range(0, len(values), size))

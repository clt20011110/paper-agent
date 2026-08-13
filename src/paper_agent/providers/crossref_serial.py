"""Crossref serial-registry venue adapter.

This adapter enumerates DOI registrations for one ISSN and year.  Its
authority is the Crossref registry, not the publisher's own article catalog.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from paper_agent.domain import EnvelopeStatus, SourceBatch, SourceEntry
from paper_agent.providers.api import (
    CrawlWindow,
    ProviderManifest,
    VenueDescriptor,
    validate_source_batch,
)


Transport = Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]


class CrossrefSerialAdapter:
    provider = "crossref_serial"

    def __init__(
        self,
        provider: str,
        transport: Transport,
        manifest: ProviderManifest,
    ) -> None:
        if provider != self.provider or manifest.provider != self.provider:
            raise ValueError("crossref_serial adapter/provider mismatch")
        self.transport = transport
        self.manifest = manifest

    def discover(
        self,
        descriptor: VenueDescriptor,
        window: CrawlWindow,
        cursor: str | None = None,
    ) -> SourceBatch:
        if descriptor.provider != self.provider:
            raise ValueError(f"{descriptor.venue_id} is not assigned to crossref_serial")
        year = int(window.year or 0)
        issns = [str(value).upper() for value in descriptor.parameters.get("issns", ())]
        registry_issn = str(descriptor.parameters.get("registry_issn") or "").upper()
        if year < 1900 or not issns:
            raise ValueError("crossref_serial requires a year and at least one ISSN")
        parameters = {
            **descriptor.parameters,
            "venue_id": descriptor.venue_id,
            "year": year,
            "date_from": window.date_from,
            "date_to": window.date_to,
            "cursor": cursor,
        }
        payload = self.transport(self.provider, "discover", parameters)
        status = EnvelopeStatus(str(payload.get("status") or "success"))
        entries = tuple(_entry(record) for record in payload.get("entries", ()))
        return validate_source_batch(SourceBatch(
            source_run_id=str(payload.get("source_run_id") or f"crossref_serial:{descriptor.venue_id}:{year}"),
            query_hash=str(payload.get("query_hash") or f"issn:{registry_issn}:{year}"),
            entries=tuple(
                replace(
                    entry,
                    metadata={
                        **entry.metadata,
                        "official_membership": False,
                        "venue_id": descriptor.venue_id,
                        "membership_authority": "crossref_registry",
                    },
                )
                for entry in entries
            ),
            next_cursor=(str(payload["next_cursor"]) if payload.get("next_cursor") else None),
            status=status,
            error=(str(payload["error"]) if payload.get("error") else None),
            raw_response_artifact_hash=(
                str(payload["raw_response_artifact_hash"])
                if payload.get("raw_response_artifact_hash")
                else None
            ),
            request_audit=tuple(
                dict(value)
                for value in payload.get("_request_audit", ())
                if isinstance(value, Mapping)
            ),
            census=(dict(payload["census"]) if isinstance(payload.get("census"), Mapping) else {}),
            warnings=tuple(str(value) for value in payload.get("warnings", ())),
        ))


def _entry(record: Mapping[str, Any]) -> SourceEntry:
    doi = str(
        record.get("doi")
        or record.get("external_id")
        or record.get("stable_id")
        or ""
    ).casefold()
    if not doi or not record.get("title"):
        raise ValueError("crossref_serial record requires DOI and title")
    return SourceEntry(
        provider="crossref_serial",
        external_id=doi,
        title=str(record["title"]),
        authors=tuple(str(value) for value in record.get("authors", ())),
        abstract=(str(record["abstract"]) if record.get("abstract") else None),
        doi=doi,
        publication_date=(
            str(record["publication_date"]) if record.get("publication_date") else None
        ),
        year=(int(record["year"]) if record.get("year") is not None else None),
        venue_name=(str(record["venue"]) if record.get("venue") else None),
        landing_url=(str(record["landing_url"]) if record.get("landing_url") else None),
        pdf_url=(str(record["pdf_url"]) if record.get("pdf_url") else None),
        metadata={
            key: value
            for key, value in record.items()
            if key not in {
                "external_id", "title", "authors", "abstract", "doi",
                "publication_date", "year", "venue", "landing_url", "pdf_url",
            }
        },
    )

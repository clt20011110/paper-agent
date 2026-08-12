"""DBLP conference table-of-contents adapter for Stage 1 censuses."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from paper_agent.domain import EnvelopeStatus, SourceBatch
from paper_agent.providers.api import (
    CrawlWindow,
    ProviderManifest,
    VenueDescriptor,
    validate_source_batch,
)
from paper_agent.providers.builtin import _source_entry


Transport = Callable[[str, str, Mapping[str, Any]], Mapping[str, Any]]


class DBLPTOCAdapter:
    provider = "dblp_toc"

    def __init__(
        self, provider: str, transport: Transport, manifest: ProviderManifest
    ) -> None:
        if provider != self.provider or manifest.provider != self.provider:
            raise ValueError("dblp_toc adapter/provider mismatch")
        self.transport = transport
        self.manifest = manifest

    def discover(
        self,
        descriptor: VenueDescriptor,
        window: CrawlWindow,
        cursor: str | None = None,
    ) -> SourceBatch:
        if descriptor.provider != self.provider:
            raise ValueError(f"{descriptor.venue_id} is not assigned to dblp_toc")
        year = int(window.year or 0)
        if year < 1900 or not descriptor.parameters.get("toc_series"):
            raise ValueError("dblp_toc requires a year and toc_series")
        parameters = {
            **descriptor.parameters,
            "venue_id": descriptor.venue_id,
            "year": year,
            "date_from": window.date_from,
            "date_to": window.date_to,
            "cursor": cursor,
        }
        payload = self.transport(self.provider, "discover", parameters)
        entries = []
        for record in payload.get("entries", ()):
            entry = _source_entry(self.provider, record)
            entries.append(
                replace(
                    entry,
                    metadata={
                        **entry.metadata,
                        "official_membership": False,
                        "venue_id": descriptor.venue_id,
                        "membership_authority": "dblp_toc",
                    },
                )
            )
        return validate_source_batch(
            SourceBatch(
                source_run_id=str(
                    payload.get("source_run_id")
                    or f"dblp_toc:{descriptor.venue_id}:{year}"
                ),
                query_hash=str(
                    payload.get("query_hash")
                    or f"toc:{descriptor.parameters['toc_series']}:{year}"
                ),
                entries=tuple(entries),
                next_cursor=(
                    str(payload["next_cursor"])
                    if payload.get("next_cursor") is not None
                    else None
                ),
                status=EnvelopeStatus(str(payload.get("status") or "success")),
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
                census=(
                    dict(payload["census"])
                    if isinstance(payload.get("census"), Mapping)
                    else {}
                ),
                warnings=tuple(str(value) for value in payload.get("warnings", ())),
            )
        )

from __future__ import annotations

import pytest

from paper_agent.domain import CitationBatch, EnvelopeStatus, SourceBatch, SourceEntry
from paper_agent.providers.api import validate_citation_batch, validate_source_batch


def test_source_batch_requires_run_query_and_structured_failure() -> None:
    batch = SourceBatch("run-1", "query-1", (), None, EnvelopeStatus.SUCCESS)
    assert validate_source_batch(batch) is batch
    with pytest.raises(ValueError, match="source_run_id"):
        validate_source_batch(SourceBatch("", "query-1", (), None, EnvelopeStatus.SUCCESS))
    with pytest.raises(ValueError, match="require an error"):
        validate_source_batch(SourceBatch("run-1", "query-1", (), None, EnvelopeStatus.FAILED))
    partial = SourceBatch("run-1", "query-1", (), None, EnvelopeStatus.PARTIAL, "one source unavailable")
    assert validate_source_batch(partial) is partial
    with pytest.raises(ValueError, match="successful"):
        validate_source_batch(SourceBatch("run-1", "query-1", (), None, EnvelopeStatus.SUCCESS, "timeout"))


def test_citation_batch_requires_structured_failure() -> None:
    batch = CitationBatch("run-1", "query-1", (), None, EnvelopeStatus.FAILED, "timeout")
    assert validate_citation_batch(batch) is batch
    partial = CitationBatch("run-1", "query-1", (), None, EnvelopeStatus.PARTIAL, "page unavailable")
    assert validate_citation_batch(partial) is partial
    with pytest.raises(ValueError, match="successful"):
        validate_citation_batch(CitationBatch("run-1", "query-1", (), None, EnvelopeStatus.SUCCESS, "timeout"))

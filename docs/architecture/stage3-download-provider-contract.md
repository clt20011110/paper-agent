# Stage 3 DownloadProvider descriptor contract

Every `DownloadProviderDescriptor` is registered with a closed contract.  The
registry rejects a missing or extra field, an unsafe side-effect boundary, and
a descriptor whose name differs from its provider implementation name.

The contract declares:

- `authentication_required`, support for main documents, supplements, and
  version selection, plus `allows_unattended`;
- `handled_domains` and `handled_resolvers`; both are checked before the
  descriptor's routing predicate;
- `retry_semantics` (`not_retryable`, `transient_retryable`, or
  `external_ledger_resumable`);
- explicit probe/fetch input and output schema IDs; and
- `idempotency_key_boundary` fixed to the persisted `FetchRequest` key, with
  `side_effect_boundary` fixed to metadata-only probe and persisted-request
  fetch.

`probe` therefore cannot download a body.  Only `fetch`, after the coordinator
has persisted and validated the `FetchRequest`, may cause a network side
effect.  New providers register a complete descriptor and contract; no central
routing branch is needed.

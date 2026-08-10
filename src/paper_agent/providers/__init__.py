"""Provider interfaces, manifests, and isolated plugin execution."""

from .api import (
    AccessPolicy,
    CredentialPolicy,
    EnrichmentResult,
    IdentityCandidate,
    ProviderManifest,
    RateLimitPolicy,
    VenueDescriptor,
    VerificationResult,
    validate_citation_batch,
    validate_source_batch,
)
from .plugins import (
    IsolatedProviderClient,
    PluginAllowlistEntry,
    PluginExecutionError,
    PluginRegistry,
    PluginRejected,
    plugin_allowlist_from_config,
)

__all__ = [
    "AccessPolicy",
    "CredentialPolicy",
    "EnrichmentResult",
    "IdentityCandidate",
    "IsolatedProviderClient",
    "PluginAllowlistEntry",
    "PluginExecutionError",
    "PluginRegistry",
    "PluginRejected",
    "ProviderManifest",
    "RateLimitPolicy",
    "VenueDescriptor",
    "VerificationResult",
    "plugin_allowlist_from_config",
    "validate_citation_batch",
    "validate_source_batch",
]

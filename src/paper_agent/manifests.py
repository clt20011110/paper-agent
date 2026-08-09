"""Versioned provider, venue, and acceptance manifest catalog."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sysconfig
from typing import Any

import yaml

from paper_agent.schema import validate


class ManifestError(ValueError):
    pass


class _ManifestLoader(yaml.SafeLoader):
    pass


for _first, _resolvers in list(_ManifestLoader.yaml_implicit_resolvers.items()):
    _ManifestLoader.yaml_implicit_resolvers[_first] = [
        item for item in _resolvers if item[0] != "tag:yaml.org,2002:timestamp"
    ]


def manifest_directory(override: Path | None = None) -> Path:
    if override:
        return override
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "providers").is_dir():
        return source_root
    return Path(sysconfig.get_path("data")) / "share" / "paper-agent"


def _load(path: Path, schema_name: str) -> dict[str, Any]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=_ManifestLoader)
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: expected a YAML object")
    validate(value, schema_name)
    return value


def _documents(root: Path, directory: str, schema_name: str) -> dict[str, dict[str, Any]]:
    paths = sorted((root / directory).glob("*.yaml"))
    if not paths:
        raise ManifestError(f"{root / directory}: no manifests found")
    documents = [_load(path, schema_name) for path in paths]
    key = "provider" if directory == "providers" else "venue_id"
    values = [document[key] for document in documents]
    if len(values) != len(set(values)):
        raise ManifestError(f"{root / directory}: duplicate {key}")
    return {document[key]: document for document in documents}


@dataclass(frozen=True, slots=True)
class ManifestCatalog:
    providers: dict[str, dict[str, Any]]
    venues: dict[str, dict[str, Any]]
    acceptances: dict[str, dict[str, Any]]

    def provider(self, provider: str) -> dict[str, Any]:
        return self.providers[provider]

    def venue(self, venue_id: str) -> dict[str, Any]:
        return self.venues[venue_id]

    def acceptance(self, venue_id: str) -> dict[str, Any]:
        return self.acceptances[venue_id]

    def runtime_venue(self, venue_id: str):
        from paper_agent.providers.api import VenueDescriptor

        venue = self.venue(venue_id)
        acceptance = self.acceptance(venue_id)
        parameters = dict(venue["provider_params"])
        if journal := acceptance.get("journal"):
            parameters.setdefault("journal_slug", journal["slug"])
            parameters.setdefault("issns", journal["issns"])
            parameters.setdefault("article_types", journal["article_types"])
        return VenueDescriptor(
            schema_version=int(venue["schema_version"]),
            venue_id=venue_id,
            provider=venue["primary_provider"],
            adapter=venue["primary_provider"],
            parameters=parameters,
        )


def load_catalog(root: Path | None = None) -> ManifestCatalog:
    directory = manifest_directory(root)
    providers = _documents(directory, "providers", "provider-manifest.schema.json")
    venues = _documents(directory, "venues", "venue-descriptor.schema.json")
    acceptances = _documents(directory, "acceptance", "acceptance-manifest.schema.json")

    if set(venues) != set(acceptances):
        raise ManifestError("venue descriptors and acceptance manifests must have identical venue_id sets")

    for venue_id, venue in venues.items():
        acceptance = acceptances[venue_id]
        primary = venue["primary_provider"]
        if primary != acceptance["primary_provider"]:
            raise ManifestError(f"{venue_id}: descriptor and acceptance primary providers differ")
        if primary not in providers:
            raise ManifestError(f"{venue_id}: unknown primary provider {primary}")
        primary_manifest = providers[primary]
        if not primary_manifest["enabled"] or "venue_primary" not in primary_manifest["roles"]:
            raise ManifestError(f"{venue_id}: primary provider must be enabled and declare venue_primary")
        if not set(acceptance["required_capabilities"]).issubset(primary_manifest["capabilities"]):
            raise ManifestError(f"{venue_id}: primary provider lacks required capabilities")
        for fallback in acceptance["fallbacks"]:
            provider = fallback["provider"]
            if provider not in providers:
                raise ManifestError(f"{venue_id}: unknown fallback provider {provider}")
            if fallback["role"] not in providers[provider]["roles"]:
                raise ManifestError(f"{venue_id}: fallback {provider} lacks role {fallback['role']}")
        fixture = (directory / acceptance["fixture_path"]).resolve()
        fixture.relative_to(directory.resolve())
        if sha256(fixture.read_bytes()).hexdigest() != acceptance["fixture_sha256"]:
            raise ManifestError(f"{venue_id}: fixture digest has drifted")

    from paper_agent.providers import builtin

    builtin_digest = sha256(Path(builtin.__file__).read_bytes()).hexdigest()
    for provider, manifest in providers.items():
        if manifest["enabled"] and manifest["builtin"] and manifest["artifact_sha256"] != builtin_digest:
            raise ManifestError(f"{provider}: built-in implementation digest has drifted")

    return ManifestCatalog(providers=providers, venues=venues, acceptances=acceptances)

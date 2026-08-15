"""Canonical venue specifications for the isolated Stage 1 package."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib
from types import MappingProxyType
from typing import NoReturn

from .errors import InputError
from .models import VenueType

__all__ = ["VenueSpec", "load_venue_spec"]


_VENUE_SPECS_DIR = Path(__file__).resolve().parent / "venue_specs"
_VENUE_ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_YEAR_PATTERN = re.compile(r"[0-9]{4}\Z")
_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "id",
        "name",
        "venue_type",
        "adapter",
        "enrichers",
        "start_year",
        "end_year",
        "held_years",
        "source",
        "year_overrides",
    }
)
_REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {"schema_version", "id", "name", "venue_type", "adapter", "enrichers", "start_year"}
)


def _schema_error(venue_id: str, message: str) -> NoReturn:
    raise InputError(f"invalid venue spec for {venue_id!r}: {message}")


def _require_text(value: object, field: str, venue_id: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _schema_error(venue_id, f"{field}: must be a non-empty string without surrounding whitespace")
    return value


def _require_year(value: object, field: str, venue_id: str) -> int:
    if type(value) is not int or not 1000 <= value <= 9999:
        _schema_error(venue_id, f"{field}: must be a non-bool four-digit integer")
    return value


def _is_applicable_year(year: int, start_year: int, end_year: int | None, held_years: tuple[int, ...] | None) -> bool:
    if type(year) is not int or not 1000 <= year <= 9999:
        return False
    if year < start_year or (end_year is not None and year > end_year):
        return False
    return held_years is None or year in held_years


def _validate_import_path(value: object, field: str, prefix: str, venue_id: str) -> str:
    path = _require_text(value, field, venue_id)
    module_name, separator, attribute = path.partition(":")
    parts = module_name.split(".")
    if (
        separator != ":"
        or len(parts) < 2
        or parts[0] != prefix
        or any(_IDENTIFIER_PATTERN.fullmatch(part) is None for part in parts[1:])
        or _IDENTIFIER_PATTERN.fullmatch(attribute) is None
    ):
        _schema_error(venue_id, f"{field}: must match {prefix}.<module>:<attribute>")
    return path


def _read_scalar_mapping(value: object, field: str, venue_id: str) -> dict[str, str | int | bool]:
    if not isinstance(value, dict):
        _schema_error(venue_id, f"{field}: must be a TOML table")
    result: dict[str, str | int | bool] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or key != key.strip():
            _schema_error(venue_id, f"{field}: keys must be non-empty strings without surrounding whitespace")
        if type(item) not in (str, int, bool):
            _schema_error(venue_id, f"{field}.{key}: value must be str, int, or bool")
        result[key] = item
    return result


@dataclass(frozen=True, slots=True)
class VenueSpec:
    """An immutable, validated description of one canonical venue."""

    venue_id: str
    name: str
    venue_type: VenueType
    adapter: str
    enrichers: tuple[str, ...]
    start_year: int
    end_year: int | None
    held_years: tuple[int, ...] | None
    source: Mapping[str, str | int | bool]
    year_overrides: Mapping[str, Mapping[str, str | int | bool]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "enrichers", tuple(self.enrichers))
        if self.held_years is not None:
            object.__setattr__(self, "held_years", tuple(self.held_years))
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))
        overrides = {
            year: MappingProxyType(dict(values))
            for year, values in self.year_overrides.items()
        }
        object.__setattr__(self, "year_overrides", MappingProxyType(overrides))

    def is_applicable(self, year: int) -> bool:
        """Return whether *year* is a declared venue year."""

        return _is_applicable_year(year, self.start_year, self.end_year, self.held_years)

    def source_for_year(self, year: int) -> Mapping[str, str | int | bool]:
        """Return a read-only copy of base source parameters resolved for *year*."""

        if not self.is_applicable(year):
            raise InputError(f"venue {self.venue_id!r} is not applicable for year {year!r}")
        resolved = dict(self.source)
        override = self.year_overrides.get(str(year))
        if override is None:
            override = self.year_overrides.get(year)  # type: ignore[arg-type]
        if override is not None:
            resolved.update(override)
        return MappingProxyType(resolved)


def _validate_document(document: object, venue_id: str) -> VenueSpec:
    if not isinstance(document, dict):
        _schema_error(venue_id, "top level must be a TOML table")

    unknown_fields = sorted(set(document) - _ALLOWED_TOP_LEVEL_FIELDS)
    if unknown_fields:
        _schema_error(venue_id, f"unknown top-level field(s): {', '.join(unknown_fields)}")
    missing_fields = sorted(_REQUIRED_TOP_LEVEL_FIELDS - set(document))
    if missing_fields:
        _schema_error(venue_id, f"missing required field(s): {', '.join(missing_fields)}")

    schema_version = document["schema_version"]
    if type(schema_version) is not int or schema_version != 1:
        _schema_error(venue_id, "schema_version: must be the non-bool integer 1")

    catalog_id = _require_text(document["id"], "id", venue_id)
    if catalog_id != venue_id:
        _schema_error(venue_id, f"id: must match requested venue ID {venue_id!r} and the file name")
    name = _require_text(document["name"], "name", venue_id)

    raw_venue_type = _require_text(document["venue_type"], "venue_type", venue_id)
    try:
        venue_type = VenueType(raw_venue_type)
    except ValueError as error:
        raise InputError(
            f"invalid venue spec for {venue_id!r}: venue_type: must be 'conference' or 'journal'"
        ) from error

    adapter = _validate_import_path(document["adapter"], "adapter", "adapters", venue_id)
    raw_enrichers = document["enrichers"]
    if not isinstance(raw_enrichers, list):
        _schema_error(venue_id, "enrichers: must be a TOML array")
    enrichers = tuple(
        _validate_import_path(item, f"enrichers[{index}]", "enrichers", venue_id)
        for index, item in enumerate(raw_enrichers)
    )
    if len(set(enrichers)) != len(enrichers):
        _schema_error(venue_id, "enrichers: entries must not be repeated")

    start_year = _require_year(document["start_year"], "start_year", venue_id)
    end_year = None
    if "end_year" in document:
        end_year = _require_year(document["end_year"], "end_year", venue_id)
        if start_year > end_year:
            _schema_error(venue_id, "start_year: must be less than or equal to end_year")

    held_years = None
    if "held_years" in document:
        raw_held_years = document["held_years"]
        if not isinstance(raw_held_years, list) or not raw_held_years:
            _schema_error(venue_id, "held_years: when present, must be a non-empty TOML array")
        held_years = tuple(
            _require_year(item, f"held_years[{index}]", venue_id)
            for index, item in enumerate(raw_held_years)
        )
        if any(previous >= current for previous, current in zip(held_years, held_years[1:])):
            _schema_error(venue_id, "held_years: must be strictly increasing without duplicates")
        if any(
            year < start_year or (end_year is not None and year > end_year)
            for year in held_years
        ):
            _schema_error(venue_id, "held_years: every year must be within start_year and end_year")

    source = _read_scalar_mapping(document.get("source", {}), "source", venue_id)
    raw_overrides = document.get("year_overrides", {})
    if not isinstance(raw_overrides, dict):
        _schema_error(venue_id, "year_overrides: must be a TOML table")
    year_overrides: dict[str, dict[str, str | int | bool]] = {}
    for raw_year, values in raw_overrides.items():
        if not isinstance(raw_year, str) or _YEAR_PATTERN.fullmatch(raw_year) is None:
            _schema_error(venue_id, "year_overrides: keys must be four-digit year strings")
        year = _require_year(int(raw_year), f"year_overrides.{raw_year}", venue_id)
        if not _is_applicable_year(year, start_year, end_year, held_years):
            _schema_error(venue_id, f"year_overrides.{raw_year}: year is not applicable")
        year_overrides[raw_year] = _read_scalar_mapping(
            values, f"year_overrides.{raw_year}", venue_id
        )

    return VenueSpec(
        venue_id=venue_id,
        name=name,
        venue_type=venue_type,
        adapter=adapter,
        enrichers=enrichers,
        start_year=start_year,
        end_year=end_year,
        held_years=held_years,
        source=source,
        year_overrides=year_overrides,
    )


def load_venue_spec(venue_id: str) -> VenueSpec:
    """Load exactly the TOML file for one canonical venue ID."""

    if not isinstance(venue_id, str) or _VENUE_ID_PATTERN.fullmatch(venue_id) is None:
        raise InputError(
            f"invalid venue ID {venue_id!r}: expected a lowercase hyphenated slug"
        )

    spec_path = _VENUE_SPECS_DIR / f"{venue_id}.toml"
    try:
        with spec_path.open("rb") as stream:
            document = tomllib.load(stream)
    except FileNotFoundError as error:
        raise InputError(f"unknown venue: {venue_id!r}") from error
    except tomllib.TOMLDecodeError as error:
        raise InputError(f"invalid TOML for venue {venue_id!r}: {error}") from error
    except UnicodeDecodeError as error:
        raise InputError(f"invalid TOML encoding for venue {venue_id!r}") from error
    except OSError as error:
        raise InputError(f"could not read venue spec for {venue_id!r}: {error}") from error

    return _validate_document(document, venue_id)

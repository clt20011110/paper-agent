"""Small exception hierarchy for the standalone Stage 1 package."""

__all__ = [
    "Stage1Error",
    "InputError",
    "CollectionError",
    "ContractError",
    "PublicationError",
]


class Stage1Error(Exception):
    """Base exception for Stage 1 operations."""


class InputError(Stage1Error):
    """Invalid user or catalog input."""


class CollectionError(Stage1Error):
    """Failure while collecting source membership."""


class ContractError(Stage1Error):
    """Violation of a Stage 1 data contract."""


class PublicationError(Stage1Error):
    """Failure while publishing Stage 1 artifacts."""

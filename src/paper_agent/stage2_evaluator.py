"""Private-side helpers for issuing Stage 2 hidden-promotion attestations.

The evaluator owns the private key and hidden evaluation material.  This module
only accepts the public-safe signing payload defined by the schema; it never
accepts labels, predictions, pair identifiers, or a private key value from an
argument, environment variable, or release bundle.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any, Mapping

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .schema import SchemaValidationError, validate
from .stage2_hidden_attestation import issue_hidden_promotion_attestation
from .stage2_search import ReleasedStage2, load_stage2_release


_MAX_PRIVATE_KEY_FILE_BYTES = 16 * 1024


class Stage2EvaluatorError(ValueError):
    """An evaluator signing input or private-key file is unsafe or invalid."""


def load_hidden_evaluator_private_key(path: Path) -> Ed25519PrivateKey:
    """Load an unencrypted PKCS#8 PEM Ed25519 key from an owner-only file.

    The evaluator operator supplies the path out of band.  The file must be a
    non-symlink regular file, owned by the current effective user, and have
    mode ``0600``.  Private key bytes are neither returned nor retained.
    """

    key_bytes = _read_owner_only_private_key_file(path)
    if not (
        key_bytes.startswith(b"-----BEGIN PRIVATE KEY-----")
        and key_bytes.rstrip().endswith(b"-----END PRIVATE KEY-----")
    ):
        raise Stage2EvaluatorError(
            "hidden evaluator private key must be an unencrypted PKCS#8 PEM file"
        )
    try:
        key = serialization.load_pem_private_key(key_bytes, password=None)
    except (TypeError, ValueError, UnsupportedAlgorithm) as error:
        raise Stage2EvaluatorError("hidden evaluator private key is not valid PEM") from error
    if not isinstance(key, Ed25519PrivateKey):
        raise Stage2EvaluatorError("hidden evaluator private key must use Ed25519")
    canonical = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    if key_bytes != canonical:
        raise Stage2EvaluatorError(
            "hidden evaluator private key must contain one canonical PKCS#8 PEM block"
        )
    return key


def issue_hidden_promotion_from_payload(
    payload: Mapping[str, Any], private_key: Ed25519PrivateKey
) -> dict[str, Any]:
    """Validate and sign the public-safe evaluator payload.

    ``private_key`` is an in-memory object loaded or provisioned by the caller;
    no function in this module accepts a private-key string or raw key bytes.
    """

    try:
        validate(payload, "stage2-hidden-evaluator-signing-input.schema.json")
    except SchemaValidationError as error:
        raise Stage2EvaluatorError(str(error)) from error
    return issue_hidden_promotion_attestation(payload, private_key)


def verify_public_stage2_release(
    release_path: Path,
    plan: Mapping[str, Any],
    *,
    hidden_trust_path: Path,
) -> ReleasedStage2:
    """Verify a v3 release with an explicitly deployment-controlled trust root."""

    return load_stage2_release(
        release_path,
        plan,
        hidden_trust_path=hidden_trust_path,
    )


def _read_owner_only_private_key_file(path: Path) -> bytes:
    try:
        initial = path.lstat()
    except OSError as error:
        raise Stage2EvaluatorError(f"cannot inspect hidden evaluator private key file: {path}") from error
    _validate_private_key_file_metadata(path, initial)

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Stage2EvaluatorError(f"cannot open hidden evaluator private key file: {path}") from error
    with os.fdopen(descriptor, "rb") as key_file:
        opened = os.fstat(key_file.fileno())
        _validate_private_key_file_metadata(path, opened)
        if _file_state(initial) != _file_state(opened):
            raise Stage2EvaluatorError("hidden evaluator private key file changed while opening")
        key_bytes = key_file.read(_MAX_PRIVATE_KEY_FILE_BYTES + 1)
        final = os.fstat(key_file.fileno())
        if _file_state(opened) != _file_state(final):
            raise Stage2EvaluatorError("hidden evaluator private key file changed while reading")
    if len(key_bytes) > _MAX_PRIVATE_KEY_FILE_BYTES:
        raise Stage2EvaluatorError("hidden evaluator private key file is too large")
    return key_bytes


def _validate_private_key_file_metadata(path: Path, details: os.stat_result) -> None:
    if not stat.S_ISREG(details.st_mode):
        raise Stage2EvaluatorError(f"hidden evaluator private key is not a regular file: {path}")
    if stat.S_IMODE(details.st_mode) != 0o600:
        raise Stage2EvaluatorError("hidden evaluator private key file must have mode 0600")
    if hasattr(os, "geteuid") and details.st_uid != os.geteuid():
        raise Stage2EvaluatorError(
            "hidden evaluator private key file must be owned by the current user"
        )


def _file_state(details: os.stat_result) -> tuple[int, ...]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )

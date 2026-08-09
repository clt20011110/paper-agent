"""Canonical JSON identities used by plans, grants, and artifacts."""

from __future__ import annotations

import hashlib
from typing import Any

import rfc8785


def canonical_json(value: Any) -> bytes:
    return rfc8785.dumps(value)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()

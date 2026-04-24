"""Regression tests for stage1 platform selection."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from paper_agent import _resolve_platform


def test_acl_uses_official_acl_adapter():
    assert _resolve_platform("ACL", "acl", "conference", {}) == "acl"


def test_aaai_openreview_alias_prefers_official_adapter():
    assert _resolve_platform("AAAI", "openreview", "conference", {}) == "aaai"


def test_ijcai_defaults_to_official_adapter():
    assert _resolve_platform("IJCAI", None, "conference", {}) == "ijcai"

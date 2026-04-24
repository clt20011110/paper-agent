"""Minimal local fallback for feedparser.parse used by arXiv adapter tests.

This module provides a tiny subset of the `feedparser` API:
- parse(content) -> object with `.entries`
- each entry exposes attributes used in this repo: id, title, summary,
  authors (with .name), tags (with .term), links, published_parsed

If the real `feedparser` package is installed, Python import resolution may still
prefer this local module due to project `PYTHONPATH`; this implementation is kept
compatible for current usage in the codebase.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List
import time
import xml.etree.ElementTree as ET


def _strip(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _child_text(node: ET.Element, name: str) -> str:
    for child in list(node):
        if _strip(child.tag) == name:
            return (child.text or "").strip()
    return ""


def parse(content: Any) -> SimpleNamespace:
    """Parse a minimal Atom feed payload into a feed-like object."""
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="ignore")
    else:
        text = str(content)

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return SimpleNamespace(entries=[])

    entries: List[SimpleNamespace] = []

    for node in root.iter():
        if _strip(node.tag) != "entry":
            continue

        entry_id = _child_text(node, "id")
        title = _child_text(node, "title")
        summary = _child_text(node, "summary")
        published = _child_text(node, "published")

        authors = []
        tags = []
        links = []

        for child in list(node):
            tag = _strip(child.tag)
            if tag == "author":
                name = _child_text(child, "name")
                if name:
                    authors.append(SimpleNamespace(name=name))
            elif tag == "category":
                term = child.attrib.get("term", "").strip()
                if term:
                    tags.append(SimpleNamespace(term=term))
            elif tag == "link":
                links.append(dict(child.attrib))

        published_parsed = None
        if published:
            try:
                # Accepts e.g. 2024-01-31T12:34:56Z
                published_parsed = time.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                published_parsed = None

        entries.append(
            SimpleNamespace(
                id=entry_id,
                title=title,
                summary=summary,
                authors=authors,
                tags=tags,
                links=links,
                published_parsed=published_parsed,
            )
        )

    return SimpleNamespace(entries=entries)

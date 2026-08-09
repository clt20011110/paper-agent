"""Minimal JSON worker used for approved third-party provider plugins."""

from __future__ import annotations

import importlib
import json
import sys
from typing import Any


def load_entry_point(value: str) -> Any:
    module_name, attribute = value.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def main(argv: list[str] | None = None) -> int:
    arguments = argv or sys.argv[1:]
    if len(arguments) != 1:
        raise SystemExit("usage: worker entry_point")
    target = load_entry_point(arguments[0])
    handler = target()
    payload = json.load(sys.stdin)
    result = handler(payload) if callable(handler) else handler.handle(payload)
    json.dump(result, sys.stdout, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

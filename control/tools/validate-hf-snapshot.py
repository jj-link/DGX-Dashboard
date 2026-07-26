#!/usr/bin/env python3
"""Fail when a Hugging Face snapshot references unavailable files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def validate_reference(index_path: Path, snapshot_path: Path, value: str) -> None:
    if not value:
        fail(f"{index_path.name!r} contains an invalid weight file reference")
    reference = PurePosixPath(value)
    if reference.is_absolute() or ".." in reference.parts:
        fail(f"{index_path.name!r} contains unsafe weight file reference {value!r}")
    target = index_path.parent.joinpath(*reference.parts)
    try:
        target.relative_to(snapshot_path)
    except ValueError:
        fail(f"{index_path.name!r} contains unsafe weight file reference {value!r}")
    if not target.is_file():
        fail(f"{index_path.name!r} references missing weight file {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot
    if not snapshot.is_dir():
        fail(f"snapshot directory {str(snapshot)!r} is unavailable")

    for entry in snapshot.rglob("*"):
        if entry.is_symlink() and not entry.exists():
            fail(f"snapshot entry {str(entry.relative_to(snapshot))!r} is a broken symlink")

    for index_path in sorted(snapshot.rglob("*.index.json")):
        if not index_path.is_file():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            fail(f"cannot read weight index {index_path.name!r}: {error}")
        if not isinstance(payload, dict) or "weight_map" not in payload:
            continue
        weight_map = payload["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            fail(f"{index_path.name!r} contains an invalid weight_map")
        references: set[str] = set()
        for reference in weight_map.values():
            if not isinstance(reference, str):
                fail(f"{index_path.name!r} contains an invalid weight file reference")
            references.add(reference)
        for reference in references:
            validate_reference(index_path, snapshot, reference)


if __name__ == "__main__":
    main()

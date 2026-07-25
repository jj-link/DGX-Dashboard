#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec python3 - "$ROOT" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys

ROOT = Path(sys.argv[1]).resolve()
MANIFEST = ROOT / "migration" / "source-manifest.tsv"
ALLOWED_DISPOSITIONS = {
    "import",
    "wsl-wins",
    "merge",
    "dependency-pin",
    "archive-only",
    "generated-ignore",
}
SOURCE_FILE_KINDS = {"source-file", "generated-file"}
SOURCE_ROOT_KINDS = {"excluded-root", "dependency-tree"}
DESTINATION_DISPOSITIONS = {"import", "wsl-wins", "merge", "dependency-pin"}
NO_DESTINATION_DISPOSITIONS = {"archive-only", "generated-ignore"}
HOSTS = {"wsl", "spark1-ts", "spark2-ts", "spark3-ts"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
LOCAL_SOURCE_ROOT = PurePosixPath("/home/workbench/inference")
REMOTE_SOURCE_ROOT = PurePosixPath("/home/jjlink/inference")
REMOTE_ARCHIVE_ROOT = PurePosixPath(
    "/home/jjlink/inference.pre-standardize-20260724-c081345"
)


def fail(message: str) -> None:
    failures.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if not MANIFEST.is_file():
    raise SystemExit(f"missing migration manifest: {MANIFEST}")

with MANIFEST.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    expected_fields = [
        "host",
        "source_path",
        "source_sha256",
        "kind",
        "classification",
        "disposition",
        "destination",
        "destination_sha256",
        "transform",
    ]
    if reader.fieldnames != expected_fields:
        raise SystemExit(
            f"invalid manifest columns: expected {expected_fields!r}, got {reader.fieldnames!r}"
        )
    rows = list(reader)

failures: list[str] = []
seen: set[tuple[str, str]] = set()
remote_sources: dict[str, list[dict[str, str]]] = {
    "spark1-ts": [],
    "spark2-ts": [],
    "spark3-ts": [],
}
local_sources: list[dict[str, str]] = []

for line_number, row in enumerate(rows, start=2):
    prefix = f"manifest line {line_number}"
    host = row["host"]
    source_path = row["source_path"]
    source_hash = row["source_sha256"]
    kind = row["kind"]
    disposition = row["disposition"]
    destination = row["destination"]
    destination_hash = row["destination_sha256"]
    transform = row["transform"]

    key = (host, source_path)
    if key in seen:
        fail(f"{prefix}: duplicate source {host}:{source_path}")
    seen.add(key)

    if host not in HOSTS:
        fail(f"{prefix}: invalid host {host!r}")
    if not source_path.startswith("/"):
        fail(f"{prefix}: source_path must be absolute: {source_path!r}")
    if kind not in SOURCE_FILE_KINDS | SOURCE_ROOT_KINDS:
        fail(f"{prefix}: invalid kind {kind!r}")
    if disposition not in ALLOWED_DISPOSITIONS:
        fail(f"{prefix}: invalid disposition {disposition!r}")
    if not row["classification"]:
        fail(f"{prefix}: classification is empty")
    if not transform:
        fail(f"{prefix}: transform is empty")

    if kind in SOURCE_FILE_KINDS:
        if not HEX64.fullmatch(source_hash):
            fail(f"{prefix}: source_sha256 must be 64 lowercase hex characters")
    elif source_hash != "-":
        fail(f"{prefix}: root source_sha256 must be '-'")

    if disposition in DESTINATION_DISPOSITIONS:
        if destination == "-":
            fail(f"{prefix}: {disposition} requires a destination")
        else:
            pure_destination = PurePosixPath(destination)
            if pure_destination.is_absolute() or ".." in pure_destination.parts:
                fail(f"{prefix}: destination must stay below the repository: {destination!r}")
            if not HEX64.fullmatch(destination_hash):
                fail(f"{prefix}: destination_sha256 must be 64 lowercase hex characters")
            else:
                destination_path = ROOT.joinpath(*pure_destination.parts)
                if not destination_path.is_file():
                    fail(f"{prefix}: destination is missing or not a file: {destination}")
                elif sha256(destination_path) != destination_hash:
                    fail(f"{prefix}: destination hash mismatch: {destination}")
    elif disposition in NO_DESTINATION_DISPOSITIONS:
        if destination != "-" or destination_hash != "-":
            fail(f"{prefix}: {disposition} must not name a destination")

    if transform in {"identity", "identity-copy", "identical-source"}:
        if source_hash != destination_hash:
            fail(f"{prefix}: {transform} source and destination hashes differ")

    source_record = {
        "line": str(line_number),
        "path": source_path,
        "kind": kind,
        "sha256": source_hash,
    }
    if host == "wsl":
        source = PurePosixPath(source_path)
        try:
            relative_source = source.relative_to(LOCAL_SOURCE_ROOT)
        except ValueError:
            relative_source = None
        # A different tracked destination is the preserved pre-migration source.
        # The original in-place path may now contain the normalized replacement.
        source_was_relocated = (
            relative_source is not None
            and destination != "-"
            and PurePosixPath(destination) != relative_source
        )
        if not source_was_relocated:
            local_sources.append(source_record)
    elif host in remote_sources:
        source = PurePosixPath(source_path)
        try:
            relative_source = source.relative_to(REMOTE_SOURCE_ROOT)
        except ValueError:
            relative_source = None
        if relative_source is not None:
            source_record["path"] = str(REMOTE_ARCHIVE_ROOT.joinpath(*relative_source.parts))
        remote_sources[host].append(source_record)


def validate_observed_source(record: dict[str, str], observed: dict[str, str], host: str) -> None:
    line = record["line"]
    path = record["path"]
    state = observed.get("state")
    if state == "missing":
        return
    if record["kind"] in SOURCE_ROOT_KINDS:
        if state != "directory":
            fail(f"manifest line {line}: source root is not a directory on {host}: {path}")
        return
    if state != "file":
        fail(f"manifest line {line}: source is not a file on {host}: {path}")
    elif observed.get("sha256") != record["sha256"]:
        fail(f"manifest line {line}: source hash mismatch on {host}: {path}")


for record in local_sources:
    path = Path(record["path"])
    if not os.path.lexists(path):
        observed = {"state": "missing"}
    elif path.is_file():
        observed = {"state": "file", "sha256": sha256(path)}
    elif path.is_dir():
        observed = {"state": "directory"}
    else:
        observed = {"state": "other"}
    validate_observed_source(record, observed, "wsl")

remote_program = r'''
import hashlib
import json
import os
import sys


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

result = []
for record in json.load(sys.stdin):
    path = record["path"]
    if not os.path.lexists(path):
        observed = {"state": "missing"}
    elif os.path.isfile(path):
        observed = {"state": "file", "sha256": sha256(path)}
    elif os.path.isdir(path):
        observed = {"state": "directory"}
    else:
        observed = {"state": "other"}
    result.append(observed)
json.dump(result, sys.stdout, separators=(",", ":"))
'''

for host, records in remote_sources.items():
    if not records:
        continue
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"python3 -c {shlex.quote(remote_program)}"],
        input=json.dumps(records, separators=(",", ":")),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        fail(
            f"source verification failed on {host}: "
            f"{completed.stderr.strip() or f'exit {completed.returncode}'}"
        )
        continue
    try:
        observed_records = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        fail(f"source verification returned invalid JSON on {host}: {error}")
        continue
    if len(observed_records) != len(records):
        fail(
            f"source verification returned {len(observed_records)} records on {host}; "
            f"expected {len(records)}"
        )
        continue
    for record, observed in zip(records, observed_records, strict=True):
        validate_observed_source(record, observed, host)

if failures:
    print("migration gate failed:", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    raise SystemExit(1)

print(
    "migration gate passed: "
    f"{len(rows)} manifest rows, "
    f"{sum(len(records) for records in remote_sources.values())} remote source records"
)
PY

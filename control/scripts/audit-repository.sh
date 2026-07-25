#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REPO_ROOT="$(cd "$CONTROL_ROOT/.." && pwd -P)"
exec python3 - "$REPO_ROOT" "$CONTROL_ROOT" <<'PY'
from __future__ import annotations

from collections import defaultdict
import csv
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(sys.argv[1]).resolve()
CONTROL_ROOT = Path(sys.argv[2]).resolve()
CONTROL_PREFIX = PurePosixPath(CONTROL_ROOT.relative_to(ROOT).as_posix())
MAX_BLOB_BYTES = 100 * 1024 * 1024
LEGACY_DASHBOARD_COMMIT = "e85268cddf8a8fefbbf6b3a8b1c4178b7e5e02c9"
WEIGHT_SUFFIXES = {
    ".safetensors",
    ".gguf",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
}
GENERATED_NAMES = {
    "candidate-process.json",
    "startup.json",
    "summary.json",
    "image.json",
    "models.json",
    "short.json",
    "long.json",
}
GENERATED_SEGMENTS = {"logs", "results", "runs"}
LOCAL_STATE_SEGMENTS = {".agents", ".claude", ".codex"}
BUILD_SEGMENTS = {
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "build",
    "dist",
    "runtime-build",
}
CACHE_SEGMENTS = {"blobs", "snapshots"}
TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
}
SECRET_NAMES = {
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
    ".bash_history",
    ".zsh_history",
}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
TOKEN_PATTERNS = {
    "GitHub token": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{20,255})"
    ),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{30,255}"),
}
PRIVATE_KEY_PATTERN = re.compile(
    rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
)


def git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", "replace").strip()
        raise SystemExit(error or f"git {' '.join(arguments)} failed")
    return completed.stdout


def decode_path(value: bytes) -> str:
    return value.decode("utf-8", "surrogateescape")


def parse_index() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for record in git("ls-files", "-s", "-z").split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_id, stage = metadata.decode("ascii").split()
        if stage != "0":
            raise SystemExit(f"unmerged index entry: {decode_path(raw_path)}")
        if mode == "160000":
            raise SystemExit(f"submodule entry is not permitted: {decode_path(raw_path)}")
        entries.append((object_id, decode_path(raw_path)))
    return entries


def parse_history() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    commits = git("rev-list", "--all").decode("ascii").splitlines()
    seen_commits: set[str] = set()
    for commit in commits:
        if not commit or commit in seen_commits:
            continue
        seen_commits.add(commit)
        tree = git("ls-tree", "-r", "-z", "--full-tree", commit)
        for record in tree.split(b"\0"):
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            if object_type == "commit" or mode == "160000":
                raise SystemExit(
                    f"submodule entry is not permitted in {commit}: {decode_path(raw_path)}"
                )
            if object_type == "blob":
                entries.append((object_id, decode_path(raw_path)))
    return entries

def parse_commit(commit: str) -> set[tuple[str, str]]:
    entries: set[tuple[str, str]] = set()
    tree = git("ls-tree", "-r", "-z", "--full-tree", commit)
    for record in tree.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split()
        if object_type == "commit" or mode == "160000":
            raise SystemExit(
                f"submodule entry is not permitted in {commit}: {decode_path(raw_path)}"
            )
        if object_type == "blob":
            entries.add((object_id, decode_path(raw_path)))
    return entries


def parse_commit_history(commit: str) -> set[tuple[str, str]]:
    entries: set[tuple[str, str]] = set()
    for revision in git("rev-list", commit).decode("ascii").splitlines():
        entries.update(parse_commit(revision))
    return entries


def dependency_targets() -> list[PurePosixPath]:
    manifest = CONTROL_ROOT / "dependencies" / "manifest.tsv"
    if not manifest.is_file():
        raise SystemExit(f"missing dependency manifest: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    targets: list[PurePosixPath] = []
    for row in rows:
        target = PurePosixPath(row["target"])
        if target.is_absolute() or ".." in target.parts:
            raise SystemExit(f"invalid dependency target: {row['target']!r}")
        targets.extend((target, CONTROL_PREFIX / target))
    return targets


def path_violation(path_text: str, targets: list[PurePosixPath]) -> str | None:
    path = PurePosixPath(path_text)
    try:
        scoped_path = path.relative_to(CONTROL_PREFIX)
    except ValueError:
        scoped_path = path
    parts = scoped_path.parts
    lower_parts = tuple(part.lower() for part in parts)
    name = scoped_path.name
    lower_name = name.lower()

    for target in targets:
        if path == target or target in path.parents:
            return f"vendored dependency path {target}"
    if parts and parts[0] == "tok-qwen35-122b":
        return "model/tokenizer cache root"
    if len(parts) >= 2 and parts[0:2] == ("benchmarks", "tokenizers"):
        return "downloaded tokenizer root"
    if any(part.startswith("models--") for part in parts):
        return "Hugging Face repository cache"
    if any(part in CACHE_SEGMENTS for part in parts):
        return "Hugging Face blob/snapshot cache"
    if name in TOKENIZER_NAMES:
        return "downloaded tokenizer asset"
    if path.suffix.lower() in WEIGHT_SUFFIXES:
        return "model weight"
    if any(part in BUILD_SEGMENTS for part in parts):
        return "downloaded or generated dependency/build tree"
    if any(part in GENERATED_SEGMENTS for part in parts):
        return "generated result tree"
    if any(part.startswith("verify-") for part in parts):
        return "generated verification tree"
    if any(part in LOCAL_STATE_SEGMENTS for part in parts):
        return "local agent/editor state"
    if lower_name in GENERATED_NAMES:
        return "generated runtime output"
    if lower_name.endswith(".log") or lower_name.endswith(".sqlite") or ".sqlite-" in lower_name:
        return "generated log/database"
    if lower_name.endswith((".tar", ".tar.gz", ".tar.xz", ".tar.zst", ".tgz")):
        return "generated archive"
    if lower_name in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES:
        return "secret/key material"
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "secret environment file"
    if len(parts) >= 2 and parts[0] == "benchmarks":
        if path.suffix.lower() in {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".html"}:
            if "config" not in lower_parts:
                return "generated benchmark output"
    return None


index_entries = parse_index()
history_entries = parse_history()
legacy_dashboard_entries = parse_commit_history(LEGACY_DASHBOARD_COMMIT)
entries = index_entries + history_entries
targets = dependency_targets()
failures: list[str] = []
object_paths: dict[str, set[str]] = defaultdict(set)

for object_id, path in index_entries:
    object_paths[object_id].add(path)
    violation = path_violation(path, targets)
    if violation:
        failures.append(f"{path}: {violation}")

for object_id, path in history_entries:
    object_paths[object_id].add(path)
    violation = path_violation(path, targets)
    if violation and (object_id, path) not in legacy_dashboard_entries:
        failures.append(f"{path}: {violation}")

process = subprocess.Popen(
    ["git", "-C", str(ROOT), "cat-file", "--batch"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
)
assert process.stdin is not None
assert process.stdout is not None

for object_id in sorted(object_paths):
    process.stdin.write(object_id.encode("ascii") + b"\n")
    process.stdin.flush()
    header = process.stdout.readline().rstrip(b"\n")
    fields = header.split()
    if len(fields) != 3 or fields[1] != b"blob":
        failures.append(f"{object_id}: expected blob, got {header.decode('ascii', 'replace')}")
        continue
    size = int(fields[2])
    content = process.stdout.read(size)
    terminator = process.stdout.read(1)
    if len(content) != size or terminator != b"\n":
        failures.append(f"{object_id}: truncated git cat-file response")
        break
    paths = sorted(object_paths[object_id])
    display_path = paths[0]
    if size > MAX_BLOB_BYTES:
        failures.append(
            f"{display_path}: blob is {size} bytes; limit is {MAX_BLOB_BYTES} bytes"
        )
    if PRIVATE_KEY_PATTERN.search(content):
        failures.append(f"{display_path}: PEM private key content")
    for label, pattern in TOKEN_PATTERNS.items():
        if pattern.search(content):
            failures.append(f"{display_path}: live {label} signature")

process.stdin.close()
process.stdout.close()
returncode = process.wait()
if returncode != 0:
    failures.append(f"git cat-file exited with status {returncode}")

if failures:
    print("repository audit failed:", file=sys.stderr)
    for failure in sorted(set(failures)):
        print(f"- {failure}", file=sys.stderr)
    raise SystemExit(1)

print(
    "repository audit passed: "
    f"{len(index_entries)} index paths, "
    f"{len(history_entries)} reachable-history paths, "
    f"{len(object_paths)} unique blobs"
)
PY

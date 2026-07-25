#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
from pathlib import Path, PurePosixPath

KEYS = (
    "IMAGE",
    "MODEL",
    "MODEL_REVISION",
    "MODEL_HOST_PATH",
    "SERVED",
    "DRAFTER",
    "DRAFTER_REVISION",
    "DRAFTER_HOST_PATH",
    "TOKENIZER",
    "TOKENIZER_REVISION",
    "TOKENIZER_HOST_PATH",
    "CONTAINER_NAME",
)
REVISION_RE = re.compile(r"[0-9a-f]{40}")
REPOSITORY_SEGMENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
REPOSITORY_RE = re.compile(rf"{REPOSITORY_SEGMENT}/{REPOSITORY_SEGMENT}")
CONTAINER_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")
IMAGE_NAME = r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
IMAGE_DIGEST_RE = re.compile(
    rf"{IMAGE_NAME}(?::[0-9]+)?(?:/{IMAGE_NAME})+@sha256:[0-9a-f]{{64}}"
)
LOCAL_IMAGE_RE = re.compile(
    rf"{IMAGE_NAME}(?:/{IMAGE_NAME})*:[a-zA-Z0-9][a-zA-Z0-9_.-]*"
)
SERVED_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+,-]*")
PATH_RE = re.compile(r"/[A-Za-z0-9._/+:-]+")


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def validate_path(label: str, value: str) -> None:
    if not PATH_RE.fullmatch(value):
        fail(f"{label} contains unsafe path characters")
    if str(PurePosixPath(value)) != value or "//" in value:
        fail(f"{label} must be a canonical absolute path")
    if any(part in {".", ".."} for part in PurePosixPath(value).parts):
        fail(f"{label} must not contain dot segments")


def validate_artifact(
    label: str,
    value: str,
    revision: str,
    host_path: str,
    *,
    optional: bool,
) -> None:
    if not value:
        if optional and not revision and not host_path:
            return
        fail(f"{label} must not be empty")
    if host_path:
        if revision:
            fail(f"{label}_REVISION must be empty when {label}_HOST_PATH is set")
        validate_path(label, value)
        validate_path(f"{label}_HOST_PATH", host_path)
        return
    if not REPOSITORY_RE.fullmatch(value):
        fail(f"{label} must be an owner/repository identifier")
    if not REVISION_RE.fullmatch(revision):
        fail(f"{label}_REVISION must be 40 lowercase hexadecimal characters")


def parse(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        fail(f"cannot read '{path}': {exc}")
    if b"\x00" in raw:
        fail("runtime metadata contains a NUL byte")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"runtime metadata is not UTF-8: {exc}")
    if "\r" in text:
        fail("runtime metadata must use LF line endings")

    lines = text.splitlines()
    if len(lines) != len(KEYS):
        fail(f"runtime metadata must contain exactly {len(KEYS)} assignments")

    values: dict[str, str] = {}
    for expected, line in zip(KEYS, lines, strict=True):
        key, separator, value = line.partition("=")
        if separator != "=" or key != expected:
            fail(f"expected assignment for {expected}")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            fail(f"{key} contains a control character")
        values[key] = value

    image = values["IMAGE"]
    if not IMAGE_DIGEST_RE.fullmatch(image):
        if not LOCAL_IMAGE_RE.fullmatch(image) or image.endswith(":latest"):
            fail("IMAGE must be an immutable digest reference or a versioned local tag")
    if not SERVED_RE.fullmatch(values["SERVED"]):
        fail("SERVED is invalid")
    if not CONTAINER_RE.fullmatch(values["CONTAINER_NAME"]):
        fail("CONTAINER_NAME is invalid")

    validate_artifact(
        "MODEL",
        values["MODEL"],
        values["MODEL_REVISION"],
        values["MODEL_HOST_PATH"],
        optional=False,
    )
    for label in ("DRAFTER", "TOKENIZER"):
        validate_artifact(
            label,
            values[label],
            values[f"{label}_REVISION"],
            values[f"{label}_HOST_PATH"],
            optional=True,
        )
    return values


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <runtime.env>", file=sys.stderr)
        return 2
    try:
        values = parse(Path(sys.argv[1]))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    output = sys.stdout.buffer
    for key in KEYS:
        output.write(key.encode("ascii") + b"\0" + values[key].encode("utf-8") + b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare an immutable local overlay for ThinkingCap DFlash serving."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "morosystems/ThinkingCap-Qwen3.6-27B-NVFP4"
REVISION = "656627c8f7ea4785413ab1e06f6ccd20bba6622f"
OVERLAY_NAME = f"thinkingcap-qwen36-27b-nvfp4-dflash-{REVISION}"


def validate_overlay(path: Path) -> None:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise RuntimeError(f"{path}/config.json has no quantization_config")
    if "kv_cache_scheme" in quantization:
        raise RuntimeError(f"{path}/config.json still enables the baked FP8 KV scheme")
    if not (path / "model.safetensors.index.json").exists():
        raise RuntimeError(f"{path} is missing model.safetensors.index.json")


def main() -> None:
    cache_root = Path(
        os.environ.get("THINKINGCAP_HF_CACHE", str(Path.home() / "models"))
    ).expanduser().resolve()
    snapshot = Path(
        snapshot_download(
            REPO_ID,
            revision=REVISION,
            cache_dir=cache_root / "hub",
        )
    )
    destination = cache_root / "serving" / OVERLAY_NAME

    if destination.exists():
        validate_overlay(destination)
        print(destination)
        return

    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    try:
        for source in snapshot.iterdir():
            target = temporary / source.name
            if source.name != "config.json":
                target.symlink_to(Path(os.path.relpath(source, temporary)))
                continue

            config = json.loads(source.read_text(encoding="utf-8"))
            quantization = config.get("quantization_config")
            if not isinstance(quantization, dict) or "kv_cache_scheme" not in quantization:
                raise RuntimeError(
                    "Pinned config no longer contains quantization_config.kv_cache_scheme"
                )
            quantization.pop("kv_cache_scheme")
            target.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        validate_overlay(temporary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(destination)


if __name__ == "__main__":
    main()

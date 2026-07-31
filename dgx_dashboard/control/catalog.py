"""Validated serving recipe catalog backed by control-plane runtime metadata."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


_ARTIFACT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_LAUNCH_PROFILE_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_PROFILE_VALUE_RE = re.compile(r"[a-z0-9_]+\Z")
_PROFILE_FIELDS = ("KV_CACHE_DTYPE", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "MTP_NUM_TOKENS")
_METADATA_KEYS = (
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
_TARGETS = ("local", "spark1", "spark2", "spark3", "cluster")
_ENGINES = ("vllm", "sglang")


class CatalogError(RuntimeError):
    """Raised when serving metadata or an adapter is missing or invalid."""


@dataclass(frozen=True)
class LaunchProfile:
    name: str
    kv_cache_dtype: str
    context_length: int
    max_sequences: int
    speculative_tokens: int

    def public(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kv_cache_dtype": self.kv_cache_dtype,
            "context_length": self.context_length,
            "max_sequences": self.max_sequences,
            "speculative_tokens": self.speculative_tokens,
        }


@dataclass(frozen=True)
class ServeRecipe:
    target: str
    engine: str
    artifact: str
    profile: str
    served: str
    container_name: str
    package: Path
    launch_profiles: tuple[LaunchProfile, ...] = ()
    default_launch_profile: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target, self.engine, self.artifact

    def public(self) -> dict[str, object]:
        return {
            "target": self.target,
            "engine": self.engine,
            "artifact": self.artifact,
            "profile": self.profile,
            "served": self.served,
            "launch_profiles": [profile.public() for profile in self.launch_profiles],
            "default_launch_profile": self.default_launch_profile,
        }

    def resolve_launch_profile(self, name: object, *, use_default: bool = False) -> str | None:
        if not self.launch_profiles:
            if name is not None:
                raise ValueError("launch_profile is not supported by this recipe")
            return None
        if name is None and use_default:
            name = self.default_launch_profile
        if not isinstance(name, str) or name not in {profile.name for profile in self.launch_profiles}:
            raise ValueError("launch_profile is not supported by this recipe")
        return name


RunParser = Callable[..., subprocess.CompletedProcess[bytes]]


class ServingCatalog:
    """Load the closed serving catalog using the control plane's strict parser."""

    def __init__(
        self,
        wrapper_root: Path,
        enabled_targets: Iterable[str],
        *,
        run_parser: RunParser = subprocess.run,
    ) -> None:
        self.wrapper_root = wrapper_root.resolve(strict=True)
        self.control_root = (self.wrapper_root / "control").resolve(strict=True)
        self._targets = tuple(enabled_targets)
        if not self._targets or any(target not in _TARGETS for target in self._targets):
            raise CatalogError("the enabled target list is invalid")
        self._run_parser = run_parser
        self._recipes = self._load()

    @property
    def targets(self) -> tuple[str, ...]:
        return self._targets

    def recipes(self) -> tuple[ServeRecipe, ...]:
        return tuple(self._recipes[key] for key in sorted(self._recipes))

    def get(self, target: str, engine: str, artifact: str) -> ServeRecipe:
        try:
            return self._recipes[(target, engine, artifact)]
        except KeyError as error:
            raise KeyError("unknown serving recipe") from error

    def public(self) -> dict[str, object]:
        return {
            "enabled": True,
            "targets": list(self._targets),
            "recipes": [recipe.public() for recipe in self.recipes()],
        }

    def _load(self) -> dict[tuple[str, str, str], ServeRecipe]:
        parser = (self.control_root / "tools" / "parse-runtime-env.py").resolve(strict=True)
        if not parser.is_file() or not os.access(parser, os.X_OK):
            raise CatalogError("the runtime metadata parser is unavailable")

        recipes: dict[tuple[str, str, str], ServeRecipe] = {}
        single_profiles = (
            ("rtx6000", ("local",)),
            ("spark", ("spark1", "spark2", "spark3")),
        )
        for engine in _ENGINES:
            for profile, targets in single_profiles:
                base = self.control_root / "serve" / engine / profile
                if not base.is_dir():
                    continue
                for package in self._packages(base):
                    if not (package / "serve.sh").is_file() or not os.access(package / "serve.sh", os.X_OK):
                        continue
                    metadata = self._parse(parser, package / "runtime.env")
                    for target in targets:
                        if target in self._targets:
                            recipe = ServeRecipe(
                                target=target,
                                engine=engine,
                                artifact=package.name,
                                profile=profile,
                                served=metadata["SERVED"],
                                container_name=metadata["CONTAINER_NAME"],
                                package=package,
                            )
                            self._insert(recipes, recipe)

            if "cluster" not in self._targets:
                continue
            base = self.control_root / "serve" / "cluster" / engine
            if not base.is_dir():
                continue
            for package in self._packages(base):
                if any(
                    not (package / f"{action}.sh").is_file() or not os.access(package / f"{action}.sh", os.X_OK)
                    for action in ("start", "status", "logs", "verify", "stop")
                ):
                    continue
                metadata = self._parse(parser, package / "runtime.env")
                launch_profiles, default_launch_profile = self._load_launch_profiles(package)
                recipe = ServeRecipe(
                    target="cluster",
                    engine=engine,
                    artifact=package.name,
                    profile="cluster",
                    served=metadata["SERVED"],
                    container_name=metadata["CONTAINER_NAME"],
                    package=package,
                    launch_profiles=launch_profiles,
                    default_launch_profile=default_launch_profile,
                )
                self._insert(recipes, recipe)

        if not recipes:
            raise CatalogError("the serving catalog contains no enabled recipes")
        missing = set(self._targets) - {recipe.target for recipe in recipes.values()}
        if missing:
            raise CatalogError(f"no serving recipes exist for enabled target(s): {', '.join(sorted(missing))}")
        return recipes

    def _packages(self, base: Path) -> list[Path]:
        canonical_base = base.resolve(strict=True)
        packages: list[Path] = []
        for candidate in base.iterdir():
            if not candidate.is_dir() or not _ARTIFACT_RE.fullmatch(candidate.name):
                continue
            package = candidate.resolve(strict=True)
            if package.parent != canonical_base or not (package / "runtime.env").is_file():
                continue
            packages.append(package)
        return sorted(packages, key=lambda path: path.name)

    def _load_launch_profiles(self, package: Path) -> tuple[tuple[LaunchProfile, ...], str | None]:
        root = package / "profiles"
        if not root.exists():
            return (), None
        if root.is_symlink() or not root.is_dir():
            raise CatalogError(f"launch profile path is unsafe for recipe {package.name}")
        default_path = root / "default"
        try:
            default = default_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise CatalogError(f"default launch profile is unreadable for recipe {package.name}") from error
        if not default.endswith("\n") or default.count("\n") != 1:
            raise CatalogError(f"default launch profile is invalid for recipe {package.name}")
        default = default.removesuffix("\n")
        if _LAUNCH_PROFILE_RE.fullmatch(default) is None:
            raise CatalogError(f"default launch profile is invalid for recipe {package.name}")

        canonical_root = root.resolve(strict=True)
        profiles: list[LaunchProfile] = []
        for candidate in sorted(root.glob("*.env"), key=lambda path: path.name):
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.resolve(strict=True).parent != canonical_root
                or _LAUNCH_PROFILE_RE.fullmatch(candidate.stem) is None
            ):
                raise CatalogError(f"launch profile file is unsafe for recipe {package.name}")
            try:
                text = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise CatalogError(f"launch profile is unreadable for recipe {package.name}") from error
            values: dict[str, str] = {}
            for line in text.splitlines():
                key, separator, value = line.partition("=")
                if separator and key in _PROFILE_FIELDS:
                    if key in values:
                        raise CatalogError(f"launch profile contains duplicate {key} for recipe {package.name}")
                    values[key] = value
            if tuple(values) != _PROFILE_FIELDS or _PROFILE_VALUE_RE.fullmatch(values["KV_CACHE_DTYPE"]) is None:
                raise CatalogError(f"launch profile metadata is invalid for recipe {package.name}")
            profiles.append(
                LaunchProfile(
                    name=candidate.stem,
                    kv_cache_dtype=values["KV_CACHE_DTYPE"],
                    context_length=self._profile_integer(values["MAX_MODEL_LEN"], "MAX_MODEL_LEN", 1, 10_000_000),
                    max_sequences=self._profile_integer(values["MAX_NUM_SEQS"], "MAX_NUM_SEQS", 1, 1024),
                    speculative_tokens=self._profile_integer(values["MTP_NUM_TOKENS"], "MTP_NUM_TOKENS", 0, 32),
                )
            )
        if not profiles or default not in {profile.name for profile in profiles}:
            raise CatalogError(f"default launch profile does not exist for recipe {package.name}")
        return tuple(profiles), default

    @staticmethod
    def _profile_integer(value: str, key: str, minimum: int, maximum: int) -> int:
        if not value.isascii() or not value.isdecimal():
            raise CatalogError(f"launch profile {key} must be an integer")
        result = int(value)
        if not minimum <= result <= maximum:
            raise CatalogError(f"launch profile {key} is outside the supported range")
        return result

    def _parse(self, parser: Path, metadata_path: Path) -> dict[str, str]:
        try:
            completed = self._run_parser(
                [str(parser), str(metadata_path)],
                cwd=self.wrapper_root,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise CatalogError(f"cannot validate runtime metadata for {metadata_path.name}") from error
        if completed.returncode != 0:
            raise CatalogError(f"runtime metadata validation failed for recipe {metadata_path.parent.name}")
        fields = completed.stdout.split(b"\0")
        if fields and fields[-1] == b"":
            fields.pop()
        if len(fields) != len(_METADATA_KEYS) * 2:
            raise CatalogError(f"runtime metadata parser returned an invalid record for {metadata_path.parent.name}")
        try:
            decoded = [field.decode("utf-8") for field in fields]
        except UnicodeDecodeError as error:
            raise CatalogError(f"runtime metadata is not UTF-8 for recipe {metadata_path.parent.name}") from error
        values = dict(zip(decoded[0::2], decoded[1::2], strict=True))
        if tuple(values) != _METADATA_KEYS:
            raise CatalogError(f"runtime metadata keys are invalid for recipe {metadata_path.parent.name}")
        return values

    @staticmethod
    def _insert(recipes: dict[tuple[str, str, str], ServeRecipe], recipe: ServeRecipe) -> None:
        if recipe.key in recipes:
            raise CatalogError("the serving catalog contains a duplicate recipe key")
        recipes[recipe.key] = recipe

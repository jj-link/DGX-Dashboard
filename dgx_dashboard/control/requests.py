"""Strict tagged request validation for dashboard operations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from dgx_dashboard.control.catalog import ServeRecipe, ServingCatalog


_LANGUAGES = {"cpp", "go", "java", "javascript", "python", "rust"}
_SERVING_ACTIONS = {"start", "stop", "verify"}
_ARTIFACT_RE = re.compile(r"[a-z0-9][a-z0-9_]*\Z")
_BENCHMARK_OPTION_KEYS = {
    "lang",
    "num_tests",
    "keywords",
    "max_tokens",
    "temperature",
    "timeout",
    "test_timeout",
    "concurrency",
    "reasoning",
    "reasoning_effort",
}
_TARGET_RESOURCES = {
    "local": frozenset({"target:local"}),
    "spark1": frozenset({"target:spark1"}),
    "spark2": frozenset({"target:spark2"}),
    "spark3": frozenset({"target:spark3"}),
    "cluster": frozenset({"target:spark2", "target:spark3"}),
}


class RequestValidationError(ValueError):
    """Raised when a mutation request does not match the closed schema."""


class UnknownRecipeError(RequestValidationError):
    """Raised when a syntactically valid serving recipe key is absent."""


@dataclass(frozen=True)
class OperationRequest:
    kind: str
    target: str
    resources: frozenset[str]
    public: Mapping[str, Any]
    recipe: ServeRecipe | None = None
    action: str | None = None
    options: Mapping[str, Any] | None = None


def _exact_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise RequestValidationError(f"{label} is missing field(s): {', '.join(sorted(missing))}")
    if unknown:
        raise RequestValidationError(f"{label} contains unknown field(s): {', '.join(sorted(unknown))}")


def _plain_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RequestValidationError(f"options.{name} must be an integer")
    if not minimum <= value <= maximum:
        raise RequestValidationError(f"options.{name} must be between {minimum} and {maximum}")
    return value


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RequestValidationError(f"options.{name} must be a number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise RequestValidationError(f"options.{name} must be between {minimum:g} and {maximum:g}")
    return result


def _target(value: Any, catalog: ServingCatalog) -> str:
    if not isinstance(value, str) or value not in catalog.targets:
        raise RequestValidationError("target is not enabled")
    return value


def validate_operation(payload: Any, catalog: ServingCatalog) -> OperationRequest:
    if not isinstance(payload, dict):
        raise RequestValidationError("request body must be a JSON object")
    kind = payload.get("kind")
    if kind == "serving":
        return _validate_serving(payload, catalog)
    if kind == "benchmark":
        return _validate_benchmark(payload, catalog)
    raise RequestValidationError("kind must be 'serving' or 'benchmark'")


def validate_persisted_operation(payload: Any, catalog: ServingCatalog) -> OperationRequest:
    """Validate durable history while allowing a removed serving recipe."""
    if isinstance(payload, dict) and payload.get("kind") == "serving":
        return _validate_serving(payload, catalog, allow_unknown_recipe=True)
    return validate_operation(payload, catalog)


def _validate_serving(
    payload: dict[str, Any],
    catalog: ServingCatalog,
    *,
    allow_unknown_recipe: bool = False,
) -> OperationRequest:
    _exact_keys(payload, {"kind", "action", "target", "engine", "artifact"}, "serving request")
    action = payload["action"]
    if not isinstance(action, str) or action not in _SERVING_ACTIONS:
        raise RequestValidationError("action must be start, stop, or verify")
    target = _target(payload["target"], catalog)
    engine = payload["engine"]
    artifact = payload["artifact"]
    if not isinstance(engine, str) or engine not in {"vllm", "sglang"}:
        raise RequestValidationError("engine must be vllm or sglang")
    if not isinstance(artifact, str) or _ARTIFACT_RE.fullmatch(artifact) is None:
        raise RequestValidationError("artifact must be a recipe name")
    try:
        recipe = catalog.get(target, engine, artifact)
    except KeyError as error:
        if not allow_unknown_recipe:
            raise UnknownRecipeError("unknown serving recipe") from error
        recipe = None
    public = {
        "kind": "serving",
        "action": action,
        "target": target,
        "engine": engine,
        "artifact": artifact,
    }
    return OperationRequest(
        kind="serving",
        target=target,
        resources=_TARGET_RESOURCES[target],
        public=public,
        recipe=recipe,
        action=action,
    )


def _validate_benchmark(payload: dict[str, Any], catalog: ServingCatalog) -> OperationRequest:
    _exact_keys(payload, {"kind", "benchmark", "target", "options"}, "benchmark request")
    if payload["benchmark"] != "oneshot":
        raise RequestValidationError("benchmark must be oneshot")
    target = _target(payload["target"], catalog)
    raw_options = payload["options"]
    if not isinstance(raw_options, dict):
        raise RequestValidationError("options must be a JSON object")
    unknown = set(raw_options) - _BENCHMARK_OPTION_KEYS
    if unknown:
        raise RequestValidationError(f"options contains unknown field(s): {', '.join(sorted(unknown))}")

    options: dict[str, Any] = {}
    if "lang" in raw_options:
        lang = raw_options["lang"]
        if lang is not None and (not isinstance(lang, str) or lang not in _LANGUAGES):
            raise RequestValidationError("options.lang must be a supported language or null")
        options["lang"] = lang
    if "num_tests" in raw_options:
        num_tests = raw_options["num_tests"]
        if num_tests != -1:
            num_tests = _plain_int(num_tests, "num_tests", 1, 100_000)
        elif isinstance(num_tests, bool):
            raise RequestValidationError("options.num_tests must be an integer")
        options["num_tests"] = num_tests
    if "keywords" in raw_options:
        keywords = raw_options["keywords"]
        if not isinstance(keywords, list) or len(keywords) > 32:
            raise RequestValidationError("options.keywords must be an array of at most 32 strings")
        normalized: list[str] = []
        for keyword in keywords:
            if (
                not isinstance(keyword, str)
                or not 1 <= len(keyword) <= 80
                or keyword != keyword.strip()
                or "," in keyword
                or any(ord(character) < 32 for character in keyword)
            ):
                raise RequestValidationError("options.keywords contains an invalid filter")
            normalized.append(keyword)
        options["keywords"] = normalized
    if "max_tokens" in raw_options:
        options["max_tokens"] = _plain_int(raw_options["max_tokens"], "max_tokens", 1, 1_048_576)
    if "temperature" in raw_options:
        options["temperature"] = _number(raw_options["temperature"], "temperature", 0.0, 2.0)
    if "timeout" in raw_options:
        options["timeout"] = _plain_int(raw_options["timeout"], "timeout", 1, 86_400)
    if "test_timeout" in raw_options:
        options["test_timeout"] = _plain_int(raw_options["test_timeout"], "test_timeout", 1, 86_400)
    if "concurrency" in raw_options:
        options["concurrency"] = _plain_int(raw_options["concurrency"], "concurrency", 1, 64)
    if "reasoning" in raw_options:
        reasoning = raw_options["reasoning"]
        if reasoning not in {"disabled", "enabled"}:
            raise RequestValidationError("options.reasoning must be disabled or enabled")
        options["reasoning"] = reasoning
    if "reasoning_effort" in raw_options:
        effort = raw_options["reasoning_effort"]
        if effort is not None and effort not in {"low", "medium", "high"}:
            raise RequestValidationError("options.reasoning_effort must be low, medium, high, or null")
        options["reasoning_effort"] = effort

    public = {
        "kind": "benchmark",
        "benchmark": "oneshot",
        "target": target,
        "options": options,
    }
    resources = _TARGET_RESOURCES[target] | {"benchmark-worker"}
    return OperationRequest(
        kind="benchmark",
        target=target,
        resources=frozenset(resources),
        public=public,
        options=options,
    )

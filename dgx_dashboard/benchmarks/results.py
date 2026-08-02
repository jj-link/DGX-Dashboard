"""Benchmark readers, cache, and persistent incremental summary index."""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator

from dgx_dashboard.config import BenchmarkSettings


INDEX_VERSION = 2
BENCHMARK_CACHE_TTL = 300
_SUMMARY_NAME = re.compile(r".*-oneshot-.*\.json\Z")
_TIMESTAMP = re.compile(r"(\d{8}-\d{6})(?:\.json)?\Z")
_LANGUAGES = ("python", "javascript", "go", "rust", "cpp", "java")


def _timestamp_from_name(name: str) -> str:
    match = _TIMESTAMP.search(name)
    return match.group(1) if match else "0"


def _safe_string(value: object, fallback: str = "") -> str:
    return value if isinstance(value, str) else fallback


def _language_counts(results: list[object]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        language = _safe_string(item.get("lang") or item.get("language"), "unknown")
        entry = counts.setdefault(language, {"passed": 0, "total": 0})
        entry["total"] += 1
        if item.get("ok") is True:
            entry["passed"] += 1
    return counts


def parse_summary(path: Path) -> dict[str, object] | None:
    """Parse one summary into compact, non-solution metadata."""

    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    results = data.get("results", [])
    if not isinstance(results, list):
        return None
    timestamp = _timestamp_from_name(path.name)

    if path.name.startswith("cross-agent-oneshot-"):
        agent_models = data.get("agent_models", {})
        model = agent_models.get("opencode", "") if isinstance(agent_models, dict) else ""
        if not isinstance(model, str) or not model.startswith("local"):
            return {
                "kind": "cross-agent",
                "timestamp": timestamp,
                "model": "",
                "languages": {},
            }
        opencode_results = [
            item
            for item in results
            if isinstance(item, dict) and item.get("agent") == "opencode"
        ]
        return {
            "kind": "cross-agent",
            "timestamp": timestamp,
            "model": model.replace("local-", "").replace("local_", ""),
            "languages": _language_counts(opencode_results),
        }

    passed = sum(1 for item in results if isinstance(item, dict) and item.get("ok") is True)
    prompt_tokens = sum(
        item.get("prompt_tokens", 0)
        for item in results
        if isinstance(item, dict) and isinstance(item.get("prompt_tokens", 0), (int, float))
    )
    completion_tokens = sum(
        item.get("completion_tokens", 0)
        for item in results
        if isinstance(item, dict) and isinstance(item.get("completion_tokens", 0), (int, float))
    )
    language = _safe_string(data.get("lang") or data.get("language"))
    languages = (
        {language: {"passed": passed, "total": len(results)}}
        if language
        else _language_counts(results)
    )
    metadata = data.get("meta")
    if not isinstance(metadata, dict):
        metadata = {}
    pass_rate = data.get("pass_rate")
    complete = (
        data.get("passed") == passed
        and data.get("total") == len(results)
        and isinstance(pass_rate, (int, float))
        and not isinstance(pass_rate, bool)
    )
    return {
        "kind": "oneshot",
        "timestamp": timestamp,
        "alias": _safe_string(data.get("alias"), path.name),
        "served": _safe_string(data.get("served"), _safe_string(data.get("model"))),
        "target": _safe_string(data.get("target")),
        "run_id": _safe_string(data.get("run_id")),
        "started": _safe_string(data.get("started"), timestamp),
        "finished": _safe_string(data.get("finished")),
        "language": language,
        "complete": complete,
        "quant": _safe_string(metadata.get("quant")),
        "reasoning": _safe_string(metadata.get("reasoning")),
        "passed": passed,
        "total": len(results),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "languages": languages,
    }


class ResultIndex:
    """Persist compact summary metadata and reparse only changed files."""

    def __init__(
        self,
        results_dir: Path,
        index_path: Path,
        *,
        parser: Callable[[Path], dict[str, object] | None] = parse_summary,
    ) -> None:
        self.results_dir = results_dir
        self.index_path = index_path
        self._parser = parser
        self._lock = threading.Lock()
        self._files: dict[str, dict[str, object]] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            with self.index_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION:
                return
            files = payload.get("files")
            if not isinstance(files, dict):
                return
            for relative, record in files.items():
                if (
                    isinstance(relative, str)
                    and isinstance(record, dict)
                    and isinstance(record.get("mtime_ns"), int)
                    and isinstance(record.get("size"), int)
                    and (record.get("summary") is None or isinstance(record.get("summary"), dict))
                ):
                    self._files[relative] = record
        except (OSError, UnicodeError, json.JSONDecodeError):
            return

    def _candidates(self) -> Iterator[Path]:
        try:
            entries = list(self.results_dir.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_file() and _SUMMARY_NAME.fullmatch(entry.name):
                yield entry

        imported = self.results_dir / "imported"
        if not imported.is_dir():
            return
        for directory, _, filenames in os.walk(imported):
            base = Path(directory)
            for filename in filenames:
                if _SUMMARY_NAME.fullmatch(filename):
                    yield base / filename

    def _persist(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "files": self._files,
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.index_path.parent,
                prefix=f".{self.index_path.name}.",
                delete=False,
            ) as stream:
                temporary_name = stream.name
                json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.index_path)
        except OSError:
            if temporary_name:
                try:
                    Path(temporary_name).unlink()
                except OSError:
                    pass

    def refresh(self) -> list[dict[str, object]]:
        with self._lock:
            self._load()
            seen: set[str] = set()
            changed = False
            for path in self._candidates():
                try:
                    stat = path.stat()
                    relative = path.relative_to(self.results_dir).as_posix()
                except (OSError, ValueError):
                    continue
                seen.add(relative)
                current = self._files.get(relative)
                if (
                    current is not None
                    and current.get("mtime_ns") == stat.st_mtime_ns
                    and current.get("size") == stat.st_size
                ):
                    continue
                self._files[relative] = {
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "summary": self._parser(path),
                }
                changed = True

            removed = set(self._files) - seen
            for relative in removed:
                del self._files[relative]
                changed = True
            if changed:
                self._persist()
            return self._summaries_unlocked()

    def summaries_for_run(self, run_id: str) -> list[dict[str, object]]:
        return [
            summary
            for summary in self.refresh()
            if summary.get("kind") == "oneshot" and summary.get("run_id") == run_id
        ]

    def _summaries_unlocked(self) -> list[dict[str, object]]:
        summaries: list[dict[str, object]] = []
        for relative, record in self._files.items():
            summary = record.get("summary")
            if not isinstance(summary, dict):
                continue
            item = dict(summary)
            item["path"] = relative
            summaries.append(item)
        return sorted(
            summaries,
            key=lambda item: (_safe_string(item.get("started")), _safe_string(item.get("timestamp"))),
            reverse=True,
        )

    @staticmethod
    def cross_agent_aggregate(summaries: list[dict[str, object]]) -> dict[str, object]:
        latest: dict[str, dict[str, object]] = {}
        for summary in summaries:
            if summary.get("kind") != "cross-agent":
                continue
            model = _safe_string(summary.get("model"))
            if not model:
                continue
            previous = latest.get(model)
            if previous is None or _safe_string(summary.get("timestamp")) > _safe_string(
                previous.get("timestamp")
            ):
                latest[model] = summary
        aggregated: dict[str, object] = {}
        for model, summary in latest.items():
            languages = summary.get("languages", {})
            if not isinstance(languages, dict):
                continue
            aggregated[model] = {
                language: {
                    "ok": counts.get("passed", 0),
                    "total": counts.get("total", 0),
                }
                for language, counts in languages.items()
                if isinstance(language, str) and isinstance(counts, dict)
            }
        return aggregated

    @staticmethod
    def oneshot_run_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
        """Aggregate completed per-language summaries into one row per dashboard run."""

        grouped: dict[str, list[dict[str, object]]] = {}
        for summary in summaries:
            run_id = _safe_string(summary.get("run_id"))
            language = _safe_string(summary.get("language"))
            if (
                summary.get("kind") != "oneshot"
                or summary.get("complete") is not True
                or not run_id
                or language not in _LANGUAGES
            ):
                continue
            grouped.setdefault(run_id, []).append(summary)

        rows: list[dict[str, object]] = []
        for run_id, run_summaries in grouped.items():
            latest_by_language: dict[str, dict[str, object]] = {}
            for summary in run_summaries:
                language = _safe_string(summary.get("language"))
                previous = latest_by_language.get(language)
                if previous is None or _safe_string(summary.get("started")) > _safe_string(
                    previous.get("started")
                ):
                    latest_by_language[language] = summary

            per_language: dict[str, dict[str, object]] = {}
            passed = 0
            total = 0
            for language in _LANGUAGES:
                summary = latest_by_language.get(language)
                if summary is None:
                    continue
                language_passed = summary.get("passed")
                language_total = summary.get("total")
                if (
                    not isinstance(language_passed, int)
                    or isinstance(language_passed, bool)
                    or not isinstance(language_total, int)
                    or isinstance(language_total, bool)
                    or language_total < 0
                    or language_passed < 0
                    or language_passed > language_total
                ):
                    continue
                passed += language_passed
                total += language_total
                per_language[language] = {
                    "ok": language_passed,
                    "total": language_total,
                    "pct": round(100 * language_passed / language_total, 1)
                    if language_total
                    else 0,
                }

            if not per_language:
                continue
            representative = max(
                latest_by_language.values(),
                key=lambda summary: _safe_string(summary.get("started")),
            )
            started_values = [
                _safe_string(summary.get("started"))
                for summary in latest_by_language.values()
                if _safe_string(summary.get("started"))
            ]
            rows.append(
                {
                    "model": _safe_string(
                        representative.get("alias"),
                        _safe_string(representative.get("served"), "unknown"),
                    ),
                    "params": "",
                    "quant": _safe_string(representative.get("quant")),
                    "per_lang": per_language,
                    "overall": round(100 * passed / total, 1) if total else 0,
                    "run_id": run_id,
                    "target": _safe_string(representative.get("target")),
                    "started": min(started_values) if started_values else "",
                    "reasoning": _safe_string(representative.get("reasoning")),
                }
            )
        return sorted(rows, key=lambda row: _safe_string(row.get("started")), reverse=True)


class BenchmarkResults:
    """Serve the legacy benchmark payload plus indexed dynamic summaries."""

    def __init__(
        self,
        settings: BenchmarkSettings,
        static_path: Path,
        *,
        clock: Callable[[], float] = time.time,
        cache_ttl: int = BENCHMARK_CACHE_TTL,
        index: ResultIndex | None = None,
    ) -> None:
        self._settings = settings
        self._static_path = static_path
        self._clock = clock
        self._cache_ttl = cache_ttl
        self.index = index or ResultIndex(settings.results_dir, settings.result_index_path)
        self._cache: dict[str, object] | None = None
        self._cache_time = 0.0
        self._lock = threading.Lock()

    @staticmethod
    def _parse_aider_stats(path: Path) -> dict[str, object]:
        data: dict[str, object] = {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return data
        for line in lines:
            line = line.strip()
            if (
                not line
                or line.startswith("─")
                or line.startswith("costs")
                or line.startswith("/")
                or line.startswith("- dirname")
            ):
                continue
            match = re.match(r"^(\w+):\s*(.+)$", line)
            if not match:
                continue
            key, value = match.group(1), match.group(2).strip()
            try:
                value = int(value) if "." not in value else float(value)
            except (ValueError, TypeError):
                pass
            data[key] = value
        return data

    def _multiturn_aider(self) -> list[dict[str, object]]:
        try:
            sweeps = sorted(self._settings.aider_benchmarks_dir.glob("*-aiderdkr-*"))
        except OSError:
            return []
        results: list[dict[str, object]] = []
        for sweep in sweeps:
            stats_path = sweep / "_stats.yml"
            if not stats_path.is_file():
                continue
            stats = self._parse_aider_stats(stats_path)
            prompt = stats.get("prompt_tokens", 0)
            completion = stats.get("completion_tokens", 0)
            if not isinstance(prompt, (int, float)):
                prompt = 0
            if not isinstance(completion, (int, float)):
                completion = 0
            results.append(
                {
                    "model": stats.get("model", "?"),
                    "pass1": stats.get("pass_rate_1", 0),
                    "pass2": stats.get("pass_rate_2", 0),
                    "total_tokens": prompt + completion,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total": stats.get("total_tests", 0),
                }
            )
        return sorted(results, key=lambda item: item["pass2"], reverse=True)

    def _load(self) -> dict[str, object]:
        try:
            with self._static_path.open(encoding="utf-8") as stream:
                static = json.load(stream)
            if not isinstance(static, dict):
                raise ValueError("benchmark_static.json must contain an object")
            summaries = self.index.refresh()
            legacy_oneshot = static.get("oneshot_table", [])
            if not isinstance(legacy_oneshot, list):
                legacy_oneshot = []
            return {
                "oneshot_table": [
                    *ResultIndex.oneshot_run_rows(summaries),
                    *legacy_oneshot,
                ],
                "multiturn_lang_table": static.get("multiturn_lang_table", []),
                "quant_comparison_table": static.get("quant_comparison_table", []),
                "comparison_table": static.get("comparison_table", []),
                "token_cost_table": static.get("token_cost_table", []),
                "multiturn_leaderboard": static.get("multiturn_leaderboard", []),
                "oneshot_leaderboard": ResultIndex.cross_agent_aggregate(summaries),
                "multiturn_summary": self._multiturn_aider(),
            }
        except Exception as error:
            print(f"Benchmark data load error: {error}", file=sys.stderr)
            return {}

    def get(self) -> dict[str, object]:
        now = self._clock()
        with self._lock:
            if self._cache is not None and now - self._cache_time < self._cache_ttl:
                return self._cache
            self._cache = self._load()
            self._cache_time = now
            return self._cache

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None
            self._cache_time = 0.0

    def run_links(self, run_id: str) -> list[dict[str, str]]:
        self.invalidate()
        links: list[dict[str, str]] = []
        for summary in self.index.summaries_for_run(run_id):
            relative = summary.get("path")
            if not isinstance(relative, str):
                continue
            token = base64.urlsafe_b64encode(relative.encode("utf-8")).decode("ascii").rstrip("=")
            label = " · ".join(
                value
                for value in (
                    _safe_string(summary.get("target")),
                    _safe_string(summary.get("started"), _safe_string(summary.get("timestamp"))),
                )
                if value
            )
            links.append(
                {
                    "label": label or _safe_string(summary.get("alias"), "oneshot result"),
                    "url": f"/api/benchmarks/results/{token}",
                }
            )
        return links

    def read_indexed_result(self, token: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        if not token or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_" for character in token):
            raise FileNotFoundError
        try:
            padded = token + "=" * (-len(token) % 4)
            relative_text = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        except (ValueError, UnicodeError):
            raise FileNotFoundError from None
        if base64.urlsafe_b64encode(relative_text.encode("utf-8")).decode("ascii").rstrip("=") != token:
            raise FileNotFoundError
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise FileNotFoundError
        indexed = {item.get("path") for item in self.index.refresh()}
        if relative.as_posix() not in indexed:
            raise FileNotFoundError
        root = self._settings.results_dir.resolve(strict=True)
        path = root.joinpath(*relative.parts).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(root):
            raise FileNotFoundError
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError("indexed result exceeds the response limit")
        return path.read_bytes()

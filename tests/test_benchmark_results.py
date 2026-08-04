"""Incremental benchmark result indexing contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dgx_dashboard.benchmarks.results import (
    INDEX_VERSION,
    BenchmarkResults,
    ResultIndex,
    parse_summary,
)
from dgx_dashboard.config import BenchmarkSettings


def write_summary(
    path: Path,
    *,
    num_tests: int = -1,
    keywords: str = "",
    passed: bool = True,
    lang: str = "python",
    run_id: str = "12345678-1234-4abc-8def-1234567890ab",
    started: str = "2026-07-25T14:00:00+00:00",
    complete: bool = True,
) -> None:
    payload = {
        "alias": "model-a",
        "served": "served-a",
        "meta": {
            "quant": "FP8",
            "reasoning": "enabled",
            "reasoning_effort": "high",
            "num_tests": num_tests,
            "keywords": keywords,
        },
        "target": "spark2",
        "run_id": run_id,
        "lang": lang,
        "started": started,
        "results": [
            {
                "ok": passed,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "solution.py": "large generated source must not enter the index",
            }
        ],
    }
    if complete:
        payload.update(
            {
                "passed": int(passed),
                "total": 1,
                "pass_rate": 100.0 if passed else 0.0,
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_result_index_parses_only_new_or_changed_summaries(tmp_path):
    results = tmp_path / "benchmark-results"
    results.mkdir()
    summary_path = results / "model-a-spark2-python-oneshot-20260725-140000.json"
    write_summary(summary_path)

    calls: list[str] = []

    def counting_parser(path: Path):
        calls.append(path.name)
        return parse_summary(path)

    index_path = tmp_path / "result-index.json"
    index = ResultIndex(results, index_path, parser=counting_parser)
    first = index.refresh()
    second = index.refresh()

    assert calls == [summary_path.name]
    assert first == second
    assert first[0]["target"] == "spark2"
    assert first[0]["run_id"] == "12345678-1234-4abc-8def-1234567890ab"
    assert first[0]["passed"] == 1
    assert "solution.py" not in index_path.read_text(encoding="utf-8")
    assert first[0]["language"] == "python"
    assert first[0]["complete"] is True

    assert first[0]["num_tests"] == -1
    assert first[0]["keywords"] == ""
    previous = summary_path.stat().st_mtime_ns
    write_summary(summary_path, passed=False)
    os.utime(summary_path, ns=(previous + 1_000_000_000, previous + 1_000_000_000))
    changed = index.refresh()

    assert calls == [summary_path.name, summary_path.name]
    assert changed[0]["passed"] == 0

    def must_not_parse(_path: Path):
        raise AssertionError("unchanged persisted summaries must not be reparsed")

    reloaded = ResultIndex(results, index_path, parser=must_not_parse)
    assert reloaded.refresh() == changed


def test_result_index_rebuilds_prior_schema_for_language_metadata(tmp_path):
    results = tmp_path / "benchmark-results"
    results.mkdir()
    summary_path = results / "model-a-local-python-oneshot-20260725-140000.json"
    write_summary(summary_path)
    stat = summary_path.stat()
    index_path = tmp_path / "result-index.json"
    index_path.write_text(
        json.dumps(
            {
                "version": INDEX_VERSION - 1,
                "files": {
                    summary_path.name: {
                        "mtime_ns": stat.st_mtime_ns,
                        "size": stat.st_size,
                        "summary": {"language": "unknown"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summaries = ResultIndex(results, index_path).refresh()

    assert summaries[0]["language"] == "python"
    assert summaries[0]["complete"] is True
    persisted = json.loads(index_path.read_text(encoding="utf-8"))
    assert persisted["version"] == INDEX_VERSION


def test_result_index_removes_deleted_summary(tmp_path):
    results = tmp_path / "benchmark-results"
    results.mkdir()
    summary_path = results / "model-a-local-python-oneshot-20260725-140000.json"
    write_summary(summary_path)
    index = ResultIndex(results, tmp_path / "result-index.json")
    assert len(index.refresh()) == 1

    summary_path.unlink()
    assert index.refresh() == []
    persisted = json.loads((tmp_path / "result-index.json").read_text(encoding="utf-8"))
    assert persisted["files"] == {}


def test_benchmark_results_preserves_legacy_payload_keys(tmp_path):
    results = tmp_path / "benchmark-results"
    results.mkdir()
    cross_agent = results / "cross-agent-oneshot-20260725-140000.json"
    cross_agent.write_text(
        json.dumps(
            {
                "agent_models": {"opencode": "local-model-a"},
                "results": [
                    {"agent": "opencode", "lang": "python", "ok": True},
                    {"agent": "opencode", "lang": "python", "ok": False},
                    {"agent": "aider", "lang": "python", "ok": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    static = tmp_path / "benchmark_static.json"
    static.write_text(
        json.dumps(
            {
                "oneshot_table": [{"model": "legacy"}],
                "multiturn_lang_table": [],
                "quant_comparison_table": [],
                "comparison_table": [],
                "token_cost_table": [],
                "multiturn_leaderboard": [],
            }
        ),
        encoding="utf-8",
    )
    settings = BenchmarkSettings(
        results_dir=results,
        aider_benchmarks_dir=tmp_path / "aider-benchmarks",
        result_index_path=tmp_path / "result-index.json",
    )

    payload = BenchmarkResults(settings, static).get()

    assert set(payload) == {
        "oneshot_table",
        "multiturn_lang_table",
        "quant_comparison_table",
        "comparison_table",
        "token_cost_table",
        "multiturn_leaderboard",
        "oneshot_leaderboard",
        "multiturn_summary",
    }
    assert payload["oneshot_table"] == [{"model": "legacy"}]
    assert payload["oneshot_leaderboard"] == {
        "model-a": {"python": {"ok": 1, "total": 2}}
    }


def test_only_full_unsampled_dashboard_run_is_added_to_oneshot_table(tmp_path):
    results = tmp_path / "benchmark-results"
    results.mkdir()
    languages = ("python", "javascript", "go", "rust", "cpp", "java")
    run_id = "12345678-1234-4abc-8def-1234567890ab"
    for index, language in enumerate(languages):
        write_summary(
            results / f"model-a-spark2-{run_id}-{language}-oneshot-20260725-14{index:04d}.json",
            passed=language != "cpp",
            run_id=run_id,
            lang=language,
            started=f"20260725-14{index:04d}",
        )

    partial_run_id = "87654321-4321-4cba-8fed-0987654321ab"
    write_summary(
        results / f"model-a-local-{partial_run_id}-python-oneshot-20260725-150000.json",
        run_id=partial_run_id,
        lang="python",
        started="20260725-150000",
    )

    sampled_run_id = "11111111-2222-4aaa-8bbb-333333333333"
    filtered_run_id = "44444444-5555-4aaa-8bbb-666666666666"
    for index, language in enumerate(languages):
        write_summary(
            results
            / f"model-a-local-{sampled_run_id}-{language}-oneshot-20260725-16{index:04d}.json",
            run_id=sampled_run_id,
            lang=language,
            started=f"20260725-16{index:04d}",
            num_tests=1,
        )
        write_summary(
            results
            / f"model-a-local-{filtered_run_id}-{language}-oneshot-20260725-17{index:04d}.json",
            run_id=filtered_run_id,
            lang=language,
            started=f"20260725-17{index:04d}",
            keywords="array",
        )

    static = tmp_path / "benchmark_static.json"
    static.write_text(
        json.dumps(
            {
                "oneshot_table": [{"model": "legacy"}],
                "multiturn_lang_table": [],
                "quant_comparison_table": [],
                "comparison_table": [],
                "token_cost_table": [],
                "multiturn_leaderboard": [],
            }
        ),
        encoding="utf-8",
    )
    settings = BenchmarkSettings(
        results_dir=results,
        aider_benchmarks_dir=tmp_path / "aider-benchmarks",
        result_index_path=tmp_path / "result-index.json",
    )

    payload = BenchmarkResults(settings, static).get()

    assert payload["oneshot_table"] == [
        {
            "model": "model-a",
            "params": "",
            "quant": "FP8",
            "per_lang": {
                "python": {"ok": 1, "total": 1, "pct": 100.0},
                "javascript": {"ok": 1, "total": 1, "pct": 100.0},
                "go": {"ok": 1, "total": 1, "pct": 100.0},
                "rust": {"ok": 1, "total": 1, "pct": 100.0},
                "cpp": {"ok": 0, "total": 1, "pct": 0.0},
                "java": {"ok": 1, "total": 1, "pct": 100.0},
            },
            "overall": 83.3,
            "run_id": run_id,
            "target": "spark2",
            "started": "20260725-140000",
            "reasoning": "enabled",
            "reasoning_effort": "high",
        },
        {"model": "legacy"},
    ]

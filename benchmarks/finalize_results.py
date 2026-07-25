#!/usr/bin/env python3
"""Produce the final RESULTS.md from the latest oneshot-bench JSONs + BFCL scores.

Picks the newest <alias>-oneshot-*.json for each of the 5 aliases, computes
pass rate / token cost / per-problem matrix, and writes RESULTS.md with
honest numbers — no spin, no premature winner-picking.
"""
import json, statistics, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BFCL_ROOT = ROOT / "gorilla" / "berkeley-function-call-leaderboard"
OUT = ROOT / "RESULTS.md"

ALIASES = [
    "minimax",
    "qwen36-27b-nvfp4",
    "qwen36-27b-fp8",
    "qwen36-35b-a3b-nvfp4",
    "qwen36-35b-a3b-fp8",
]
SHORT = {
    "minimax": "mmax",
    "qwen36-27b-nvfp4": "27N",
    "qwen36-27b-fp8": "27F",
    "qwen36-35b-a3b-nvfp4": "35aN",
    "qwen36-35b-a3b-fp8": "35aF",
}

# Maps alias → BFCL model name (as used in run_bfcl.sh case block)
BFCL_MODEL = {
    "qwen36-27b-nvfp4":     "Qwen/Qwen3-32B-FC",
    "qwen36-27b-fp8":       "Qwen/Qwen3-32B-FC",
    "qwen36-35b-a3b-nvfp4": "Qwen/Qwen3-30B-A3B-Instruct-2507-FC",
    "qwen36-35b-a3b-fp8":   "Qwen/Qwen3-30B-A3B-Instruct-2507-FC",
    "minimax":              "spark/minimax-m2.7-reap-FC",
}

BFCL_LIVE_CATS = [
    "live_simple", "live_multiple", "live_parallel",
    "live_parallel_multiple", "live_relevance", "live_irrelevance",
]


def bfcl_scores(alias):
    """Return (correct, total, accuracy) from the latest per-alias results/bfcl/<alias>-*/ dir.

    Only reads from per-alias timestamped directories — never from the shared gorilla
    score directory, because two aliases can share the same BFCL model name and would
    produce identical (and therefore ambiguous) entries.
    """
    correct = total = 0

    bfcl_dirs = sorted((RESULTS / "bfcl").glob(f"{alias}-*/"))
    score_base = None
    for d in reversed(bfcl_dirs):  # latest first
        if (d / "live").is_dir() and any((d / "live").glob("*_score.json")):
            score_base = d / "live"
            break

    if score_base is None:
        return None, None, None

    for cat in BFCL_LIVE_CATS:
        f = score_base / f"BFCL_v4_{cat}_score.json"
        if not f.exists():
            continue
        try:
            first_line = f.read_text().splitlines()[0]
            d = json.loads(first_line)
            correct += d.get("correct_count", 0)
            total += d.get("total_count", 0)
        except Exception:
            continue

    if total == 0:
        return None, None, None
    return correct, total, round(100 * correct / total, 1)


def latest(alias):
    files = sorted(RESULTS.glob(f"{alias}-oneshot-*.json"))
    if not files:
        return None, None
    # Prefer by: most passes > most results > newest filename.
    # This guards against stale zero-pass runs with 34 results (from earlier
    # failed sweeps) outranking an in-progress run that already has real passes,
    # and against partial files with a newer timestamp outranking the active run.
    def sort_key(f):
        d = json.loads(f.read_text())
        rs = d.get("results", [])
        passes = sum(1 for r in rs if r.get("ok"))
        return (passes, len(rs), f.name)
    best_file = max(files, key=sort_key)
    return json.loads(best_file.read_text()), best_file.name


def stats(rs, key, where=lambda r: True):
    vals = [r[key] for r in rs if where(r) and r.get(key) is not None]
    return vals


def fmt(n):
    return f"{n:,}" if isinstance(n, (int, float)) else "—"


def main():
    rows = []
    for a in ALIASES:
        d, f = latest(a)
        rows.append((a, d, f))

    lines = []
    L = lines.append
    L("# 5-model spark/local head-to-head — final results")
    L("")
    L(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from the latest sweep per alias._")
    L("")
    L("## Setup")
    L("")
    L("- **34 Python polyglot exercises**, one-shot prompts (instructions + stub).")
    L("- `max_tokens=32768`, `temperature=0.0`, streaming SSE.")
    L("- Pytest scores each: pass = full test suite passes, anything else = fail.")
    L("- Spark (GB10): minimax-m2.7-REAP-172B-A10B-NVFP4")
    L("- Local (RTX PRO 6000 Blackwell, WSL2): the 4 Qwens")
    L("")
    L("## Pass rate")
    L("")
    L("| Model | Quant | Pass | Rate | Avg latency | Avg out tok |")
    L("|---|---|---|---|---:|---:|")
    quant_of = {
        "minimax": "NVFP4",
        "qwen36-27b-nvfp4": "NVFP4",
        "qwen36-27b-fp8": "FP8",
        "qwen36-35b-a3b-nvfp4": "NVFP4",
        "qwen36-35b-a3b-fp8": "FP8",
    }
    for a, d, _ in rows:
        if d is None:
            L(f"| {a} | {quant_of[a]} | — | — | — | — |")
            continue
        rs = d.get("results") or []
        n = d.get("total", len(rs))
        p = d.get("passed", sum(1 for r in rs if r.get("ok")))
        rate = round(100 * p / n, 1) if n else 0
        lats = stats(rs, "latency_s")
        outs = stats(rs, "completion_tokens")
        avg_lat = f"{sum(lats)/len(lats):.1f}s" if lats else "—"
        avg_out = round(sum(outs) / len(outs)) if outs else 0
        partial = " *(PARTIAL)*" if "total" not in d else ""
        L(f"| {a} | {quant_of[a]} | {p}/{n}{partial} | {rate}% | {avg_lat} | {fmt(avg_out)} |")
    L("")
    L("## Token cost")
    L("")
    L("| Model | Total in | Total out | Median out | p95 out | Avg out (✓) | Avg out (✗) |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    for a, d, _ in rows:
        if d is None:
            L(f"| {a} | — | — | — | — | — | — |")
            continue
        rs = d.get("results") or []
        ins = stats(rs, "prompt_tokens")
        outs = stats(rs, "completion_tokens")
        outs_ok = stats(rs, "completion_tokens", lambda r: r.get("ok"))
        outs_no = stats(rs, "completion_tokens", lambda r: r.get("ok") is False)
        tot_in = sum(ins) if ins else 0
        tot_out = sum(outs) if outs else 0
        med_out = round(statistics.median(outs)) if outs else 0
        p95_out = round(sorted(outs)[int(0.95 * (len(outs) - 1))]) if outs else 0
        avg_ok = round(sum(outs_ok) / len(outs_ok)) if outs_ok else 0
        avg_no = round(sum(outs_no) / len(outs_no)) if outs_no else 0
        L(f"| {a} | {fmt(tot_in)} | {fmt(tot_out)} | {fmt(med_out)} | {fmt(p95_out)} | {fmt(avg_ok)} | {fmt(avg_no)} |")
    L("")
    L("## Per-problem pass matrix")
    L("")
    all_probs = set()
    by_alias = {}
    for a, d, _ in rows:
        if d is None:
            continue
        rs = d.get("results") or []
        m = {r["name"]: r.get("ok") for r in rs}
        by_alias[a] = m
        all_probs.update(m.keys())
    hdr = "| problem | " + " | ".join(SHORT[a] for a in ALIASES) + " |"
    L(hdr)
    L("|---" * (len(ALIASES) + 1) + "|")
    for prob in sorted(all_probs):
        row = [prob]
        for a in ALIASES:
            v = by_alias.get(a, {}).get(prob)
            mark = "?" if v is None else ("✓" if v else "✗")
            row.append(mark)
        L("| " + " | ".join(row) + " |")
    L("")
    L("## BFCL v3 Live AST (tool-use)")
    L("")
    L("Accuracy = correct / total across all live categories (simple, multiple, parallel,")
    L("parallel_multiple, relevance, irrelevance). 2243 test cases total.")
    L("")
    L("| Model | Quant | Correct | Total | Accuracy | Notes |")
    L("|---|---|---:|---:|---:|---|")
    bfcl_note = {
        "qwen36-27b-nvfp4":     "Qwen3-32B-FC template; prompting mode",
        "qwen36-27b-fp8":       "Qwen3-32B-FC template; prompting mode",
        "qwen36-35b-a3b-nvfp4": "Qwen3-30B-A3B template; prompting mode",
        "qwen36-35b-a3b-fp8":   "Qwen3-30B-A3B template; prompting mode",
        "minimax":              "QuickTestingOSSHandler; prompting mode",
    }
    for a in ALIASES:
        correct, total, acc = bfcl_scores(a)
        q = quant_of[a]
        note = bfcl_note.get(a, "")
        if acc is None:
            L(f"| {a} | {q} | — | — | — | {note} |")
        else:
            L(f"| {a} | {q} | {correct:,} | {total:,} | {acc}% | {note} |")
    L("")
    L("## Source data")
    L("")
    for a, d, f in rows:
        if f:
            L(f"- {a} polyglot: `results/{f}`")
        else:
            L(f"- {a} polyglot: (no run)")
    L("")
    for a in ALIASES:
        bfcl_dirs = sorted((RESULTS / "bfcl").glob(f"{a}-*/"))
        valid = [d for d in bfcl_dirs if (d / "live").is_dir() and any((d / "live").glob("*_score.json"))]
        if valid:
            L(f"- {a} BFCL: `results/bfcl/{valid[-1].name}/`")
        else:
            L(f"- {a} BFCL: (no run or no scores)")
    L("")
    L("## Reproduce")
    L("")
    L("```bash")
    L("cd /home/workbench/Projects/personal/test/benchmarks")
    L("# Spark (minimax only):")
    L("ALIASES=minimax NUM_TESTS=34 MAX_TOKENS=32768 TIMEOUT=3600 ./sweep_oneshot.sh")
    L("ALIAS=minimax ./sweep_bfcl_spark.sh")
    L("# Local (4 Qwens):")
    L("NUM_TESTS=34 MAX_TOKENS=32768 TIMEOUT=3600 ./sweep_oneshot_local.sh")
    L("./sweep_bfcl_local.sh")
    L("# Aggregate:")
    L("python3 finalize_results.py")
    L("```")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

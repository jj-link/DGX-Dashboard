#!/usr/bin/env python3
"""Collect the latest oneshot-bench result for each alias and print a
comparison table. Also emits a token-cost breakdown and a per-problem
pass matrix.

Usage:
  python aggregate_minimal.py                # all sections
  python aggregate_minimal.py --section table|tokens|matrix
"""
import argparse, json, statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ALIASES = [
    "minimax",
    "qwen36-27b-nvfp4",
    "qwen36-27b-fp8",
    "qwen36-35b-a3b-nvfp4",
    "qwen36-35b-a3b-fp8",
]


def load_latest(alias: str):
    files = sorted(RESULTS.glob(f"{alias}-oneshot-*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text()), files[-1].name


def fmt_int(n):
    return f"{n:,}" if isinstance(n, (int, float)) else "—"


def section_table(rows):
    print(f"{'alias':<26} | {'pass':>7} | {'rate':>6} | {'avg_lat_s':>10} | {'avg_out_tok':>11} | notes")
    print("-" * 95)
    for alias, data, fname in rows:
        if data is None:
            print(f"{alias:<26} | {'—':>7} | {'—':>6} | {'—':>10} | {'—':>11} | (no run)")
            continue
        rs = data.get("results") or []
        n = data.get("total", len(rs))
        p = data.get("passed", sum(1 for r in rs if r.get("ok")))
        rate = data.get("pass_rate", round(100 * p / n, 1) if n else 0)
        lats = [r["latency_s"] for r in rs if r.get("latency_s") is not None]
        outs = [r["completion_tokens"] for r in rs if r.get("completion_tokens") is not None]
        avg_lat = round(sum(lats) / len(lats), 1) if lats else 0
        avg_out = round(sum(outs) / len(outs)) if outs else 0
        errs = sum(1 for r in rs if r.get("error"))
        partial = "total" not in data
        bits = []
        if errs:
            bits.append(f"{errs} errors")
        if partial:
            bits.append("PARTIAL")
        print(f"{alias:<26} | {p:>3}/{n:<3} | {rate:>5}% | {avg_lat:>10} | {avg_out:>11} | {', '.join(bits)}")


def section_tokens(rows):
    """Token-cost breakdown: total/median/p95 input + output, and avg-by-outcome."""
    print(f"{'alias':<26} | {'tot_in':>10} | {'tot_out':>10} | {'med_out':>8} | {'p95_out':>8} | {'avg_out(✓)':>11} | {'avg_out(✗)':>11}")
    print("-" * 105)
    for alias, data, fname in rows:
        if data is None:
            print(f"{alias:<26} | (no run)")
            continue
        rs = data.get("results") or []
        ins = [r["prompt_tokens"] for r in rs if r.get("prompt_tokens") is not None]
        outs = [r["completion_tokens"] for r in rs if r.get("completion_tokens") is not None]
        outs_ok = [r["completion_tokens"] for r in rs if r.get("ok") and r.get("completion_tokens") is not None]
        outs_no = [r["completion_tokens"] for r in rs if r.get("ok") is False and r.get("completion_tokens") is not None]
        tot_in = sum(ins) if ins else 0
        tot_out = sum(outs) if outs else 0
        med_out = round(statistics.median(outs)) if outs else 0
        p95_out = round(sorted(outs)[int(0.95 * (len(outs) - 1))]) if outs else 0
        avg_ok = round(sum(outs_ok) / len(outs_ok)) if outs_ok else 0
        avg_no = round(sum(outs_no) / len(outs_no)) if outs_no else 0
        print(f"{alias:<26} | {fmt_int(tot_in):>10} | {fmt_int(tot_out):>10} | {fmt_int(med_out):>8} | {fmt_int(p95_out):>8} | {fmt_int(avg_ok):>11} | {fmt_int(avg_no):>11}")


def section_matrix(rows):
    """Per-problem pass matrix across all 5 models."""
    all_problems = set()
    by_alias = {}
    for alias, data, _ in rows:
        if data is None:
            continue
        rs = data.get("results") or []
        m = {r["name"]: r.get("ok") for r in rs}
        by_alias[alias] = m
        all_problems.update(m.keys())
    if not all_problems:
        print("(no problems run)")
        return
    short_aliases = {
        "minimax": "mmax",
        "qwen36-27b-nvfp4": "27N",
        "qwen36-27b-fp8": "27F",
        "qwen36-35b-a3b-nvfp4": "35aN",
        "qwen36-35b-a3b-fp8": "35aF",
    }
    keys = [short_aliases.get(a, a) for a, *_ in rows if a in by_alias]
    hdr = f"{'problem':<22} | " + " | ".join(f"{k:>4}" for k in keys)
    print(hdr)
    print("-" * len(hdr))
    for prob in sorted(all_problems):
        marks = []
        for alias, _, _ in rows:
            if alias not in by_alias:
                continue
            v = by_alias[alias].get(prob)
            mark = "?" if v is None else ("✓" if v else "✗")
            marks.append(f"{mark:>4}")
        print(f"{prob:<22} | " + " | ".join(marks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=["table", "tokens", "matrix", "all"], default="all")
    args = ap.parse_args()

    rows = []
    for alias in ALIASES:
        loaded = load_latest(alias)
        if loaded is None:
            rows.append((alias, None, None))
        else:
            data, fname = loaded
            rows.append((alias, data, fname))

    if args.section in ("table", "all"):
        print("\n=== pass rate ===")
        section_table(rows)
    if args.section in ("tokens", "all"):
        print("\n=== token cost ===")
        section_tokens(rows)
    if args.section in ("matrix", "all"):
        print("\n=== per-problem pass matrix ===")
        section_matrix(rows)


if __name__ == "__main__":
    main()

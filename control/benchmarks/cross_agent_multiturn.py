#!/usr/bin/env python3
"""Multi-turn cross-agent polyglot benchmark.

Runs each agent in the same exercise workdir for up to --turns attempts. After
each failed test run, the next agent invocation receives the test output and
continues editing the same files.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cross_agent_bench import (
    AGENTS,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_EST_TOKENS_PER_REQ,
    DEFAULT_LANGS,
    RESULTS_DIR,
    clean_copy,
    live_model,
    parse_num_tests,
    preflight,
    prompt_for,
    resolve_concurrency,
    select_problems,
    tqdm,
)
from oneshot_bench import LANGS, read_problem, run_in_docker


def retry_prompt(test_res: dict, turn: int) -> str:
    stdout = (test_res.get("stdout") or "").strip()
    stderr = (test_res.get("stderr") or "").strip()
    parts = [
        f"Turn {turn - 1} did not pass the tests.",
        "Continue editing the existing solution file(s) in place so the tests pass.",
        "Do not modify tests or metadata.",
        f"Test return code: {test_res.get('rc')}",
    ]
    if stdout:
        parts.append(f"\nTest stdout:\n```\n{stdout[-4000:]}\n```")
    if stderr:
        parts.append(f"\nTest stderr:\n```\n{stderr[-4000:]}\n```")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="aider,opencode",
                    help="Comma-separated: aider,opencode")
    ap.add_argument("--langs", default=DEFAULT_LANGS,
                    help="Comma-separated languages")
    ap.add_argument("--num-tests", type=parse_num_tests, default="all",
                    help="Per-language task count: positive integer or 'all'")
    ap.add_argument("--keywords", default="",
                    help="Comma-separated exercise name filters")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--api-key", default=DEFAULT_API_KEY)
    ap.add_argument("--served-model", default="",
                    help="Model id for Aider; defaults to /v1/models")
    ap.add_argument("--aider-model", default="",
                    help="Aider model name; default openai/<served-model>")
    ap.add_argument("--opencode-model", default=os.environ.get("OPENCODE_MODEL", ""),
                    help="opencode model name; default local-vllm/<served-model>")
    ap.add_argument("--concurrency", default=os.environ.get("CONCURRENCY", "auto"),
                    help="Concurrent exercise/agent tasks: auto or positive integer")
    ap.add_argument("--est-tokens-per-req", type=int, default=DEFAULT_EST_TOKENS_PER_REQ,
                    help="Token estimate for auto concurrency")
    ap.add_argument("--turns", type=int, default=3,
                    help="Maximum agent/test attempts per exercise")
    ap.add_argument("--agent-timeout", type=int, default=900)
    ap.add_argument("--test-timeout", type=int, default=300)
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--start-task", type=int, default=1,
                    help="1-indexed task offset after language/keyword selection")
    ap.add_argument("--max-tasks", type=int, default=0,
                    help="Maximum selected tasks to run after --start-task; 0 means all remaining")
    args = ap.parse_args()
    if args.est_tokens_per_req < 1:
        raise SystemExit("[fatal] --est-tokens-per-req must be positive")
    if args.turns < 1:
        raise SystemExit("[fatal] --turns must be positive")

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    unknown = [a for a in agents if a not in AGENTS]
    if unknown:
        raise SystemExit(f"[fatal] unknown agents: {', '.join(unknown)}")
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    bad_langs = [l for l in langs if l not in LANGS]
    if bad_langs:
        raise SystemExit(f"[fatal] unknown languages: {', '.join(bad_langs)}")

    preflight(langs, agents)

    served = args.served_model or live_model(args.base_url)
    agent_models = {
        "aider": args.aider_model or f"openai/{served}",
        "opencode": args.opencode_model or f"local-vllm/{served}",
    }

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    tasks = select_problems(langs, keywords, args.num_tests)
    if args.start_task < 1:
        raise SystemExit("[fatal] --start-task must be >= 1")
    start_idx = args.start_task - 1
    end_idx = None if args.max_tasks <= 0 else start_idx + args.max_tasks
    tasks = tasks[start_idx:end_idx]
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"cross-agent-multiturn-{ts}"
    out_dir = Path(args.out_dir)
    work_root = out_dir / run_name
    work_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{run_name}.json"

    summary = {
        "run": run_name,
        "mode": "multiturn",
        "turns": args.turns,
        "served_model": served,
        "agents": agents,
        "agent_models": {a: agent_models[a] for a in agents},
        "langs": langs,
        "num_tests": args.num_tests,
        "keywords": keywords,
        "concurrency": args.concurrency,
        "est_tokens_per_req": args.est_tokens_per_req,
        "results": [],
    }

    total = len(tasks) * len(agents)
    workers, kv, server_max_concurrency = resolve_concurrency(
        args.concurrency, total, args.est_tokens_per_req
    )
    summary["resolved_concurrency"] = workers
    summary["kv_cache_tokens"] = kv
    summary["server_max_concurrency"] = server_max_concurrency
    kv_text = f" KV={kv}tok/{args.est_tokens_per_req}est" if kv else ""
    server_text = f" server_max={server_max_concurrency:.2f}x" if server_max_concurrency else ""
    print(
        f"[cross-mt] agents={','.join(agents)} langs={','.join(langs)} "
        f"tasks={len(tasks)} total={total} turns={args.turns} "
        f"concurrency={workers} ({args.concurrency}){kv_text}{server_text}"
    )
    print(f"[cross-mt] work={work_root}")

    work_items = [
        (idx, agent, lang, name, src_dir)
        for idx, (lang, name, src_dir) in enumerate(tasks, start=1)
        for agent in agents
    ]

    def process_item(item):
        idx, agent, lang, name, src_dir = item
        task_dir = work_root / agent / lang / name
        clean_copy(src_dir, task_dir)
        prob = read_problem(task_dir, lang)
        turns = []
        test_res = None
        ok = False
        for turn in range(1, args.turns + 1):
            message = prompt_for(prob, lang) if turn == 1 else retry_prompt(test_res, turn)
            agent_res = AGENTS[agent](
                task_dir, prob, message, agent_models[agent],
                args.base_url, args.api_key, args.agent_timeout,
            )
            test_res = run_in_docker(prob, lang, args.test_timeout)
            ok = bool(test_res["ok"])
            turns.append({
                "turn": turn,
                "ok": ok,
                "agent_rc": agent_res["rc"],
                "agent_duration_s": agent_res["duration_s"],
                "agent_timeout": bool(agent_res.get("timeout")),
                "test": test_res,
                "agent_stdout_tail": agent_res["stdout_tail"],
                "agent_stderr_tail": agent_res["stderr_tail"],
            })
            if ok:
                break
        return {
            "index": idx,
            "agent": agent,
            "agent_model": agent_models[agent],
            "lang": lang,
            "name": name,
            "ok": ok,
            "turns_used": len(turns),
            "turns": turns,
            "test": test_res,
            "dir": str(task_dir),
        }

    started = time.perf_counter()
    done = 0
    passed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_item, item) for item in work_items]
        completed = as_completed(futures)
        progress = tqdm(completed, total=total) if tqdm else completed
        for fut in progress:
            row = fut.result()
            done += 1
            if row["ok"]:
                passed += 1
            summary["results"].append(row)
            summary["results"].sort(key=lambda r: (r["index"], agents.index(r["agent"])))
            summary_path.write_text(json.dumps(summary, indent=2))
            if tqdm:
                progress.set_postfix(
                    pass_rate=f"{passed}/{done}",
                    last=f"{row['agent']}:{row['lang']}/{row['name']}@{row['turns_used']}",
                )
            else:
                elapsed = int(time.perf_counter() - started)
                eta = int(elapsed * (total - done) / done) if done else 0
                print(f"[cross-mt] {done}/{total} pass={passed}/{done} elapsed={elapsed}s eta={eta}s", flush=True)

    by_agent = {}
    for agent in agents:
        rows = [r for r in summary["results"] if r["agent"] == agent]
        passed = sum(1 for r in rows if r["ok"])
        by_agent[agent] = {
            "passed": passed,
            "total": len(rows),
            "pass_rate": round(100 * passed / len(rows), 1) if rows else 0,
            "avg_turns_used": round(sum(r["turns_used"] for r in rows) / len(rows), 2) if rows else 0,
        }
    summary["summary"] = by_agent
    summary_path.write_text(json.dumps(summary, indent=2))
    for agent, stats in by_agent.items():
        print(
            f"[summary] {agent}: {stats['passed']}/{stats['total']} = "
            f"{stats['pass_rate']}% avg_turns={stats['avg_turns_used']}"
        )
    print(f"[cross-mt] summary={summary_path}")


if __name__ == "__main__":
    main()

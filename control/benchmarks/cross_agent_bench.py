#!/usr/bin/env python3
"""Single-shot cross-agent polyglot benchmark.

Runs the same Exercism/polyglot tasks through different coding harnesses
(initially aider and opencode), then grades the edited files once with the
same Docker test commands used by oneshot_bench.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from oneshot_bench import LANGS, POLYGLOT, read_problem, run_in_docker

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path(os.environ.get(
    "BENCHMARK_RESULTS_ROOT", "/var/lib/dgx-dashboard/benchmark-results"
))
AIDER_ROOT = SCRIPT_DIR / "aider"
AIDER_PY = AIDER_ROOT / ".venv" / "bin" / "python"

DEFAULT_LANGS = "python,javascript,go,rust,cpp,java"
DEFAULT_BASE_URL = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
DEFAULT_EST_TOKENS_PER_REQ = int(os.environ.get("EST_TOKENS_PER_REQ", "32768"))


def live_model(base_url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(f"{base_url}/models", timeout=5) as r:
        return json.loads(r.read().decode())["data"][0]["id"]


def live_vllm_capacity() -> tuple[int | None, float | None]:
    try:
        ps = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=^vllm-"],
            capture_output=True, text=True, timeout=5,
        )
        cid = ps.stdout.strip().splitlines()[0] if ps.stdout.strip() else ""
        if not cid:
            return None, None
        logs = subprocess.run(
            ["docker", "logs", cid],
            capture_output=True, text=True, timeout=10,
        ).stdout + subprocess.run(
            ["docker", "logs", cid],
            capture_output=True, text=True, timeout=10,
        ).stderr
    except Exception:
        return None, None
    import re

    matches = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", logs)
    kv = int(matches[-1].replace(",", "")) if matches else None
    max_conc_matches = re.findall(
        r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x",
        logs,
    )
    max_conc = float(max_conc_matches[-1]) if max_conc_matches else None
    return kv, max_conc


def resolve_concurrency(value: str, total: int, est_tokens_per_req: int):
    total = total if total else 1
    if value != "auto":
        n = int(value)
        return max(1, min(n, total)), None, None
    kv, server_max_concurrency = live_vllm_capacity()
    if kv:
        n = max(1, int(kv / est_tokens_per_req))
        if server_max_concurrency:
            n = min(n, max(1, int(server_max_concurrency)))
        return min(n, total), kv, server_max_concurrency
    return min(total, 1), None, server_max_concurrency


def clean_copy(src: Path, dst: Path):
    def ignore(_dir, names):
        ignored = {
            ".git", ".pytest_cache", "__pycache__", "node_modules", "target",
            "build", ".gradle", ".aider.chat.history.md", ".aider.input.history",
            ".aider.results.json", ".oneshot.results.json",
        }
        return [n for n in names if n in ignored or n.endswith(".pyc")]

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def prompt_for(prob: dict, lang: str) -> str:
    cfg = LANGS[lang]
    files = "\n".join(f"- {rel}" for rel in prob["sol_rels"])
    return (
        f"You are editing an Exercism {cfg['pretty']} exercise.\n\n"
        "Implement the solution file(s) so the unit tests pass.\n\n"
        f"Files to edit:\n{files}\n\n"
        "Instructions:\n\n"
        f"{prob['instructions']}\n\n"
        "Rules:\n"
        "- Edit the existing solution file(s) in place.\n"
        "- Do not rename files, functions, classes, or public APIs required by tests.\n"
        "- Do not modify tests or metadata.\n"
        "- Stop when the implementation is complete."
    )


def run_aider(task_dir: Path, prob: dict, message: str, model: str,
              base_url: str, api_key: str, timeout: int) -> dict:
    env = os.environ.copy()
    env["AIDER_SUPPRESS_TOKEN_LIMIT_WARNINGS"] = "1"
    env["PYTHONPATH"] = (
        f"{AIDER_ROOT}:{AIDER_ROOT / 'benchmark'}"
        f"{':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else ''}"
    )
    msg_file = task_dir / ".cross_agent_prompt.md"
    msg_file.write_text(message)
    cmd = [
        str(AIDER_PY), "-m", "aider",
        "--model", model,
        "--openai-api-base", base_url,
        "--openai-api-key", api_key,
        "--edit-format", "whole",
    ]
    model_settings_file = env.get("AIDER_MODEL_SETTINGS_FILE")
    if model_settings_file:
        cmd.extend(["--model-settings-file", model_settings_file])
    model_metadata_file = env.get("AIDER_MODEL_METADATA_FILE")
    if model_metadata_file:
        cmd.extend(["--model-metadata-file", model_metadata_file])
    cmd += [
        "--no-git",
        "--yes-always",
        "--no-auto-commits",
        "--no-dirty-commits",
        "--no-analytics",
        "--no-show-model-warnings",
        "--no-check-model-accepts-settings",
        "--no-detect-urls",
        "--stream",
        "--message-file", str(msg_file),
    ] + prob["sol_rels"]
    return run_cmd(cmd, task_dir, env, timeout, label=f"aider {task_dir.name}")


def run_opencode(task_dir: Path, prob: dict, message: str, model: str,
                 _base_url: str, _api_key: str, timeout: int) -> dict:
    cmd = [
        "opencode", "run",
        message,
        "--model", model,
        "--dir", str(task_dir),
        "--dangerously-skip-permissions",
    ]
    for rel in prob["sol_rels"]:
        cmd.extend(["--file", rel])
    return run_cmd(cmd, task_dir, os.environ.copy(), timeout, label=f"opencode {task_dir.name}")


def suppress_aider_token_limit_popup(text: str) -> str:
    """Remove aider's noisy token-limit warning block from benchmark-captured output."""
    marker = "Model openai/gemma-4-12b-it-UD-Q4_K_XL.gguf has hit a token limit!"
    while marker in text:
        marker_pos = text.find(marker)
        start = text.rfind("\n\n", 0, marker_pos)
        start = 0 if start < 0 else start
        url = "https://aider.chat/docs/troubleshooting/token-limits.html"
        end = text.find(url, marker_pos)
        if end < 0:
            break
        end += len(url)
        while end < len(text) and text[end] in "\r\n":
            end += 1
        text = text[:start] + "\n[aider token-limit warning suppressed by benchmark harness]\n" + text[end:]
    return text


def run_cmd(cmd: list, cwd: Path, env: dict, timeout: int, label: str = "agent") -> dict:
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            stdout, stderr = proc.communicate()
        elapsed = time.perf_counter() - t0
        stdout = suppress_aider_token_limit_popup(stdout)
        stderr = suppress_aider_token_limit_popup(stderr)
        if timed_out:
            return {
                "rc": None,
                "duration_s": round(elapsed, 1),
                "timeout": True,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            }
        return {
            "rc": proc.returncode,
            "duration_s": round(elapsed, 1),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "rc": None,
            "duration_s": round(elapsed, 1),
            "error": f"{type(e).__name__}: {e}",
            "stdout_tail": "",
            "stderr_tail": "",
        }


AGENTS = {
    "aider": run_aider,
    "opencode": run_opencode,
}


def parse_num_tests(value: str):
    if value.lower() == "all":
        return None
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--num-tests must be a positive integer or 'all'")
    if n < 1:
        raise argparse.ArgumentTypeError("--num-tests must be a positive integer or 'all'")
    return n


def select_problems(langs: list, keywords: list, num_tests):
    selected = []
    for lang in langs:
        root = POLYGLOT / lang / "exercises" / "practice"
        if not root.is_dir():
            raise SystemExit(f"[fatal] missing language corpus: {root}")
        count = 0
        for src_dir in sorted(root.iterdir()):
            if not src_dir.is_dir():
                continue
            if keywords and not any(k.lower() in src_dir.name.lower() for k in keywords):
                continue
            prob = read_problem(src_dir, lang)
            if not prob:
                continue
            selected.append((lang, src_dir.name, src_dir))
            count += 1
            if num_tests is not None and count >= num_tests:
                break
    return selected


def preflight(langs: list, agents: list):
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise SystemExit("[fatal] docker not available")
    for lang in langs:
        img = LANGS[lang]["image"]
        if subprocess.run(["docker", "image", "inspect", img],
                          capture_output=True).returncode != 0:
            raise SystemExit(
                f"[fatal] image {img} not found. Build it first:\n"
                f"        bash {SCRIPT_DIR / 'docker' / 'build.sh'} {lang}"
            )
    if "aider" in agents and not AIDER_PY.exists():
        raise SystemExit(f"[fatal] aider venv python missing: {AIDER_PY}")
    if "opencode" in agents and shutil.which("opencode") is None:
        raise SystemExit("[fatal] opencode not found on PATH")


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
                    help="Concurrent agent tasks: auto or positive integer")
    ap.add_argument("--est-tokens-per-req", type=int, default=DEFAULT_EST_TOKENS_PER_REQ,
                    help="Token estimate for auto concurrency")
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
    run_name = f"cross-agent-oneshot-{ts}"
    out_dir = Path(args.out_dir)
    work_root = out_dir / run_name
    work_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{run_name}.json"

    summary = {
        "run": run_name,
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
    print(f"[cross] agents={','.join(agents)} langs={','.join(langs)} tasks={len(tasks)} total={total} concurrency={workers} ({args.concurrency}){kv_text}{server_text}")
    print(f"[cross] work={work_root}")

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
        message = prompt_for(prob, lang)
        agent_res = AGENTS[agent](
            task_dir, prob, message, agent_models[agent],
            args.base_url, args.api_key, args.agent_timeout,
        )
        test_res = run_in_docker(prob, lang, args.test_timeout)
        return {
            "index": idx,
            "agent": agent,
            "agent_model": agent_models[agent],
            "lang": lang,
            "name": name,
            "ok": test_res["ok"],
            "agent_rc": agent_res["rc"],
            "agent_duration_s": agent_res["duration_s"],
            "agent_timeout": bool(agent_res.get("timeout")),
            "test": test_res,
            "agent_stdout_tail": agent_res["stdout_tail"],
            "agent_stderr_tail": agent_res["stderr_tail"],
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
                progress.set_postfix(pass_rate=f"{passed}/{done}", last=f"{row['agent']}:{row['lang']}/{row['name']}")
            else:
                elapsed = int(time.perf_counter() - started)
                eta = int(elapsed * (total - done) / done) if done else 0
                print(f"[cross] {done}/{total} pass={passed}/{done} elapsed={elapsed}s eta={eta}s", flush=True)

    by_agent = {}
    for agent in agents:
        rows = [r for r in summary["results"] if r["agent"] == agent]
        passed = sum(1 for r in rows if r["ok"])
        by_agent[agent] = {
            "passed": passed,
            "total": len(rows),
            "pass_rate": round(100 * passed / len(rows), 1) if rows else 0,
        }
    summary["summary"] = by_agent
    summary_path.write_text(json.dumps(summary, indent=2))
    for agent, stats in by_agent.items():
        print(f"[summary] {agent}: {stats['passed']}/{stats['total']} = {stats['pass_rate']}%")
    print(f"[cross] summary={summary_path}")


if __name__ == "__main__":
    main()

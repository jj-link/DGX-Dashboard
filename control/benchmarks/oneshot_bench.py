#!/usr/bin/env python3
"""Oneshot polyglot bench (multi-language, Docker-isolated tests).

For each polyglot exercise in the chosen language:
  1. Build a single-shot prompt with the exercise instructions + stub file.
  2. POST to the OpenAI-compatible endpoint with max_tokens cap.
  3. Extract the code block from the response (strip <think> blocks).
  4. Stage the extracted code in a temporary exercise copy.
  5. Run that language's test command INSIDE A DOCKER CONTAINER, so the
     toolchain (pytest/jest/go/cargo/gradle/g++) lives in the image and
     never has to be installed on the host.
  6. Record pass/fail/error.

Output:
  - per-problem .oneshot.results.json next to each exercise
  - a summary JSON at results/<alias>-<target>-<run-id>-<lang>-oneshot-<ts>.json

Usage:
  python oneshot_bench.py <alias> --lang rust [--num-tests N]
                          [--keywords W,...] [--timeout SEC] [--concurrency K]
"""
import argparse, hashlib, json, os, platform, re, shlex, shutil, subprocess, sys, tempfile, threading, time, uuid
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = os.environ.get("OPENAI_API_BASE", "http://gx10-a6c7.lan:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
BENCHMARK_TARGET = "direct"
BENCHMARK_RUN_ID = str(uuid.uuid4())

SCRIPT_DIR = Path(__file__).resolve().parent
POLYGLOT = Path(os.environ.get(
    "POLYGLOT_ROOT",
    str(SCRIPT_DIR / "aider" / "tmp.benchmarks" / "polyglot-benchmark"),
))
RESULTS_DIR = Path(os.environ.get(
    "BENCHMARK_RESULTS_ROOT", "/var/lib/dgx-dashboard/benchmark-results"
))

# Per-language config. `test_cmd(sol, test)` returns the shell command run
# inside the container, cwd=/work (the mounted exercise dir). `sol`/`test`
# are the solution / first-test filenames from .meta/config.json.
# Images are prebuilt by docker/build.sh with toolchain + deps baked in, so
# `test_cmd` only runs tests — never installs anything (no network at run).
LANGS = {
    "python": {
        "pretty": "Python", "fence": "python", "image": "polybench-python:latest",
        "test_cmd": lambda sol, test: f"python -m pytest -x -q {shlex.quote(test)}",
    },
    "javascript": {
        "pretty": "JavaScript", "fence": "javascript", "image": "polybench-javascript:latest",
        # deps baked at /node_modules (NODE_PATH set in image); jest resolves
        # them via upward module resolution from the mounted /work.
        "test_cmd": lambda sol, test: "/node_modules/.bin/jest ./*",
    },
    "go": {
        "pretty": "Go", "fence": "go", "image": "polybench-go:latest",
        "test_cmd": lambda sol, test: "go test ./...",
    },
    "rust": {
        "pretty": "Rust", "fence": "rust", "image": "polybench-rust:latest",
        # crate cache prewarmed in CARGO_HOME; cached crates aren't re-fetched.
        "test_cmd": lambda sol, test: "cargo test --quiet",
    },
    "java": {
        "pretty": "Java", "fence": "java", "image": "polybench-java:latest",
        # gradle cache (plugins + junit/assertj) baked -> fully offline.
        "test_cmd": lambda sol, test: "gradle test --no-daemon -q --console=plain --offline",
    },
    "cpp": {
        "pretty": "C++", "fence": "cpp", "image": "polybench-cpp:latest",
        # exercism C++ bundles Catch2 (test/tests-main.cpp). Compile
        # solution + test + Catch2 main directly — no cmake needed.
        "test_cmd": lambda sol, test: (
            f"g++ -std=c++17 -I. {shlex.quote(sol)} {shlex.quote(test)} "
            f"test/tests-main.cpp -o /tmp/run && /tmp/run"
        ),
    },
}

SYS_TMPL = (
    "You are an expert {pretty} programmer. You will be given an exercise with "
    "an instructions document and a {pretty} stub file. Your job is to implement "
    "the function(s)/class(es) in the stub so the unit tests pass.\n\n"
    "Output the COMPLETE, FINAL contents of the stub file in a single {fence} "
    "code fence (```{fence} ... ```). No prose outside the fence. Do not "
    "rename functions, classes, or the file — only implement them."
)

USER_TMPL = (
    "# Instructions\n\n{instructions}\n\n"
    "# Stub file: {fname}\n\n```{fence}\n{stub}\n```\n\n"
    "Output the complete final {fname} file in a single ```{fence}``` block."
)

# Multi-file variants (e.g. C++ needs both <name>.cpp and <name>.h). The
# model must emit each file as a `FILE: <path>` line followed by one fenced
# block, so the response can be split back into per-file contents.
SYS_TMPL_MULTI = (
    "You are an expert {pretty} programmer. You will be given an exercise with "
    "an instructions document and {n} {pretty} stub files. Implement them so "
    "the unit tests pass.\n\n"
    "Output the COMPLETE, FINAL contents of EVERY file. For each file, output "
    "a line of exactly `FILE: <path>` (the path as given) on its own line, "
    "immediately followed by a single ```{fence} ... ``` code block with that "
    "file's full contents. No other prose. Do not rename anything."
)

USER_TMPL_MULTI = (
    "# Instructions\n\n{instructions}\n\n# Stub files\n\n{stub_blocks}\n\n"
    "Output every file, each preceded by its own `FILE: <path>` line and a "
    "single ```{fence}``` block."
)


def read_problem(prob_dir: Path, lang: str):
    """Resolve instructions + the solution/test files via .meta/config.json
    (uniform across all six languages). Returns None if the exercise is
    missing instructions or a usable solution file."""
    instr_path = prob_dir / ".docs" / "instructions.md"
    if not instr_path.exists():
        return None
    instructions = instr_path.read_text()
    append_path = prob_dir / ".docs" / "instructions.append.md"
    if append_path.exists():
        instructions += "\n\n# Instructions append\n\n" + append_path.read_text()

    cfg_path = prob_dir / ".meta" / "config.json"
    sol_files, test_files = [], []
    if cfg_path.exists():
        try:
            files = json.loads(cfg_path.read_text()).get("files", {})
            sol_files = files.get("solution", []) or []
            test_files = files.get("test", []) or []
        except json.JSONDecodeError:
            pass
    sol_files = [f for f in sol_files if (prob_dir / f).exists()]
    test_files = [f for f in test_files if (prob_dir / f).exists()]
    if not sol_files or not test_files:
        return None
    return {
        "instructions": instructions,
        "sol_rels": sol_files,                   # ALL files the model must fill
        "stubs": {f: (prob_dir / f).read_text() for f in sol_files},
        "sol_rel": sol_files[0],                 # primary, for logging
        "stub_path": prob_dir / sol_files[0],
        "test_rel": test_files[0],
        "dir": prob_dir,
    }


def chat(model: str, messages: list, max_tokens: int, timeout_s: float,
         temperature: float = 1.0, enable_thinking: bool = False,
         reasoning_effort: str | None = None):
    """Streaming chat completion, assembled into one response. Streaming
    avoids urlopen waiting for the whole body — important for reasoning
    models decoding at <5 tok/s where a long answer exceeds the timeout."""
    body = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "top_p": 0.95, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if enable_thinking:
        template_kwargs = {"enable_thinking": True}
        if reasoning_effort:
            template_kwargs["reasoning_effort"] = reasoning_effort
        body["chat_template_kwargs"] = template_kwargs
    elif os.environ.get("ONESHOT_DISABLE_THINKING") == "1":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"},
    )
    t0 = time.perf_counter()
    content_parts, reasoning_parts = [], []
    finish_reason, usage = None, None
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            c = delta.get("content")
            if c:
                content_parts.append(c)
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                reasoning_parts.append(r)
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr
    elapsed = time.perf_counter() - t0
    return {
        "choices": [{
            "message": {"content": "".join(content_parts),
                        "reasoning_content": "".join(reasoning_parts)},
            "finish_reason": finish_reason,
        }],
        "usage": usage or {},
    }, elapsed


def extract_code(text: str, fence: str) -> str:
    """Pull the most-likely code block from a model response. Priority:
    language-tagged closed > language-tagged open > generic closed > generic
    open. (Earlier minimax bug: model echoed instructions in a closed generic
    fence then opened a truncated language fence; the longest-closed heuristic
    picked the instructions. Tag-specific first avoids that.)"""
    if not text:
        return ""
    finders = [
        re.compile(rf"```(?:{fence})\s*\n(.*?)\n```", re.DOTALL),
        re.compile(rf"```(?:{fence})\s*\n(.*)", re.DOTALL),
        re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL),
        re.compile(r"```[^\n]*\n(.*)", re.DOTALL),
    ]
    for finder in finders:
        matches = finder.findall(text)
        if matches:
            return max(matches, key=len).strip()
    return ""


def extract_files(text: str, fence: str, sol_rels: list) -> dict:
    """Split a multi-file response into {sol_rel: code}.

    Single-file solutions defer to extract_code (unchanged behavior for
    python/js/go/rust/java). For multi-file (C++ .cpp/.h) we walk every
    fenced block and attribute it to whichever expected file is named in the
    text just before it (the `FILE: <path>` marker, or the path/basename
    appearing anywhere in the preamble). If nothing matches but the block
    count equals the file count, fall back to positional assignment."""
    if len(sol_rels) == 1:
        code = extract_code(text, fence)
        return {sol_rels[0]: code} if code else {}
    if not text:
        return {}
    block_re = re.compile(r"```[^\n]*\n(.*?)\n```", re.DOTALL)
    out, blocks, prev = {}, [], 0
    for m in block_re.finditer(text):
        preamble = text[prev:m.start()]
        blocks.append((preamble, m.group(1).strip()))
        prev = m.end()
    for preamble, code in blocks:
        # Prefer the file whose path/basename appears latest in the preamble.
        best, best_pos = None, -1
        for rel in sol_rels:
            for needle in (rel, rel.split("/")[-1]):
                pos = preamble.rfind(needle)
                if pos > best_pos:
                    best, best_pos = rel, pos
        if best is not None and best not in out:
            out[best] = code
    missing = [r for r in sol_rels if r not in out]
    if missing and len(blocks) == len(sol_rels):
        for rel, (_, code) in zip(sol_rels, blocks):  # positional fallback
            out.setdefault(rel, code)
    return {k: v for k, v in out.items() if v}


def run_in_docker(prob: dict, lang: str, timeout: int, solutions: dict):
    """Test generated solutions without modifying the benchmark corpus.

    A temporary exercise copy receives the generated source. It is mounted
    read-only at /src and copied to container-local /work before testing, so
    neither candidate source nor root-owned build artifacts reach the corpus.
    """
    cfg = LANGS[lang]
    test = cfg["test_cmd"](prob["sol_rel"], prob["test_rel"])
    inner = f"mkdir -p /work && cp -a /src/. /work/ && cd /work && {test}"
    cname = f"bench-{lang}-{prob['dir'].name}-{os.getpid()}-{threading.get_ident()}"

    with tempfile.TemporaryDirectory(prefix=f"oneshot-{lang}-") as tmp:
        staged = Path(tmp) / prob["dir"].name
        shutil.copytree(prob["dir"], staged)
        for rel, src in solutions.items():
            (staged / rel).write_text(src)
        cmd = [
            "docker", "run", "--rm", "--name", cname,
            "--label", f"io.dgx-dashboard.kind=benchmark",
            "--label", f"io.dgx-dashboard.run-id={BENCHMARK_RUN_ID}",
            "-v", f"{staged}:/src:ro",
            # `bash -c`, NOT `-lc`: a login shell re-sources /etc/profile and
            # can reset PATH, dropping toolchain directories.
            cfg["image"], "bash", "-c", inner,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
            return {
                "ok": proc.returncode == 0, "rc": proc.returncode,
                "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-1000:],
            }
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", cname],
                           capture_output=True, text=True)
            return {"ok": False, "rc": None, "stdout": "",
                    "stderr": f"TIMEOUT after {timeout}s"}


def result_alias(alias: str | None, served: str) -> str:
    """Return an explicit legacy label or a filename-safe served-model label."""
    if alias:
        return alias
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", served).strip("._-")
    return safe or "model"

def benchmark_identity(environment: dict[str, str] | None = None) -> tuple[str, str]:
    """Return a validated target and collision-free run identifier."""
    values = os.environ if environment is None else environment
    target = values.get("DGX_DASHBOARD_BENCHMARK_TARGET", "direct")
    if target not in {"direct", "local", "spark1", "spark2", "spark3", "cluster"}:
        raise ValueError(f"invalid benchmark target {target!r}")
    run_id = values.get("DGX_DASHBOARD_RUN_ID") or str(uuid.uuid4())
    try:
        parsed = uuid.UUID(run_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("DGX_DASHBOARD_RUN_ID must be a lowercase UUID") from error
    if str(parsed) != run_id:
        raise ValueError("DGX_DASHBOARD_RUN_ID must be a lowercase UUID")
    return target, run_id


def main():
    global BENCHMARK_TARGET, BENCHMARK_RUN_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("alias", nargs="?")
    ap.add_argument("--lang", default=None, choices=sorted(LANGS),
                    help="Polyglot language to benchmark (default: all languages)")
    ap.add_argument("--served-model", help="Override served-model-name")
    ap.add_argument("--num-tests", type=int, default=-1, help="-1 = all")
    ap.add_argument("--keywords", default="", help="Comma-separated name filters")
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="Output token budget. MUST be large for reasoning "
                         "models — they spend 15-30k tokens thinking before "
                         "emitting code; 8192 truncates them mid-reasoning "
                         "and floors the score with empty answers.")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature (default: 1.0)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-request HTTP timeout (s)")
    ap.add_argument("--test-timeout", type=int, default=300,
                    help="Per-exercise Docker test timeout (s); higher than "
                         "before since first run also pulls the image / deps")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Problems in parallel (1=serial). vLLM batches the "
                         "model calls; Docker isolates the test runs.")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--backend", default=None,
                    help="Serving backend: vllm, sglang, llama.cpp, etc.")
    ap.add_argument("--quant", default=None,
                    help="Quantization: fp8, nvfp4, fp16, awq, etc.")
    ap.add_argument("--kv-cache-type", default=None,
                    help="KV cache dtype: nvfp4_ds_mla, fp8, auto, etc.")
    ap.add_argument("--spec-decode", default=None,
                    help="Speculative decoding: dspark, dflash, eagle3, none, etc.")
    ap.add_argument("--hardware", default=None,
                    help="Hardware description: e.g. '2x GB10 Spark', 'RTX PRO 6000'")
    ap.add_argument("--context-length", type=int, default=None,
                    help="Max model context length (tokens)")
    ap.add_argument("--tp-size", type=int, default=None,
                    help="Tensor parallel size")
    ap.add_argument("--reasoning", default="disabled",
                    help="Reasoning mode: disabled, enabled (default: disabled)")
    ap.add_argument("--reasoning-effort", choices=("low", "medium", "high"),
                    help="Model reasoning effort passed through chat_template_kwargs")
    ap.add_argument("--engine-version", default=None,
                    help="Exact inference-engine version or build identifier")
    ap.add_argument("--runtime-image", default=None,
                    help="Serving container image name and tag")
    ap.add_argument("--runtime-image-digest", default=None,
                    help="Immutable serving container image ID or digest")
    ap.add_argument("--model-source", default=None,
                    help="Model repository or weight source")
    ap.add_argument("--model-revision", default=None,
                    help="Immutable model revision or weight identifier")
    args = ap.parse_args()
    try:
        args.target, args.run_id = benchmark_identity()
    except ValueError as error:
        ap.error(str(error))
    BENCHMARK_TARGET = args.target
    BENCHMARK_RUN_ID = args.run_id

    try:
        with urllib.request.urlopen(f"{BASE_URL}/models", timeout=5) as r:
            m = json.loads(r.read().decode())["data"][0]["id"]
            print(f"[health] served model: {m}", flush=True)
            served = args.served_model or m
            args.alias = result_alias(args.alias, served)
    except Exception as e:
        print(f"[health] FAIL: {e}", file=sys.stderr)
        sys.exit(2)

    # Docker preflight — after endpoint health, before any grading container.
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        print("[fatal] docker not available (docker info failed)", file=sys.stderr)
        sys.exit(2)

    langs_to_run = [args.lang] if args.lang else sorted(LANGS)

    for lang in langs_to_run:
        run_one_language(lang, args, served)

def detect_gpu():
    """Detect which GPU is serving the model by finding the busiest GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        # Find GPU with most memory in use (likely the serving one)
        gpu_mem = []
        for line in out.stdout.strip().split("\n"):
            parts = line.split(", ")
            if len(parts) == 2:
                idx = parts[0].strip()
                mem = int(re.findall(r'(\d+)', parts[1].strip())[0])
                gpu_mem.append((idx, mem))
        if gpu_mem:
            best_gpu = max(gpu_mem, key=lambda x: x[1])
            if best_gpu[1] > 1000:  # Only count if >1GB in use
                name_out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", best_gpu[0]],
                    capture_output=True, text=True, timeout=10,
                )
                if name_out.returncode == 0:
                    return name_out.stdout.strip()
    except Exception:
        pass
    return None


def command_output(cmd):
    """Return stripped stdout for a best-effort metadata probe."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else None
    except Exception:
        return None


def run_one_language(lang, args, served):
    """Run oneshot benchmark for a single language."""
    ex_root = POLYGLOT / lang / "exercises" / "practice"
    if not ex_root.is_dir():
        print(f"[fatal] no exercises for lang={lang}: {ex_root}", file=sys.stderr)
        sys.exit(2)

    # The prebuilt image must already exist locally (built by docker/build.sh).
    img = LANGS[lang]["image"]
    if subprocess.run(["docker", "image", "inspect", img],
                       capture_output=True).returncode != 0:
        print(f"[fatal] image {img} not found. Build it first:\n"
              f"        bash {SCRIPT_DIR / 'docker' / 'build.sh'} {lang}",
              file=sys.stderr)
        sys.exit(2)
    print(f"[docker] using prebuilt image {img}", flush=True)
    image_digest = command_output(
        ["docker", "image", "inspect", "--format={{.Id}}", img])

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    problems = []
    for d in sorted(ex_root.iterdir()):
        if not d.is_dir():
            continue
        if keywords and not any(k.lower() in d.name.lower() for k in keywords):
            continue
        p = read_problem(d, lang)
        if p:
            problems.append((d.name, p))
    if args.num_tests > 0:
        problems = problems[: args.num_tests]
    print(f"[plan] {len(problems)} {lang} problems alias={args.alias} model={served}",
          flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    summary_path = out_dir / (
        f"{args.alias}-{args.target}-{args.run_id}-{lang}-oneshot-{ts}.json"
    )
    gpu = detect_gpu()
    summary = {"alias": args.alias, "target": args.target, "run_id": args.run_id,
               "lang": lang, "served": served, "started": ts, "gpu": gpu,
               "results": []}
    # Record serving metadata for reproducibility
    meta = {
        "target": args.target,
        "run_id": args.run_id,
        "backend": args.backend,
        "quant": args.quant,
        "kv_cache_type": args.kv_cache_type,
        "spec_decode": args.spec_decode,
        "hardware": args.hardware,
        "context_length": args.context_length,
        "tp_size": args.tp_size,
        "endpoint": os.environ.get("OPENAI_API_BASE", "unknown"),
        "reasoning": args.reasoning,
        "chat_template_enable_thinking": (
            True if args.reasoning == "enabled" else
            False if os.environ.get("ONESHOT_DISABLE_THINKING") == "1" else None
        ),
        "reasoning_effort": args.reasoning_effort,
        "engine_version": args.engine_version,
        "runtime_image": args.runtime_image,
        "runtime_image_digest": args.runtime_image_digest,
        "model_source": args.model_source,
        "model_revision": args.model_revision,
        "temperature": args.temperature,
        "top_p": 0.95,
        "max_tokens": args.max_tokens,
        "request_timeout_s": args.timeout,
        "test_timeout_s": args.test_timeout,
        "concurrency": args.concurrency,
        "num_tests": args.num_tests,
        "keywords": args.keywords,
        "test_image": img,
        "test_image_digest": image_digest,
        "benchmark_corpus_revision": command_output(
            ["git", "-C", str(POLYGLOT), "rev-parse", "HEAD"]),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "harness_python": platform.python_version(),
    }
    summary["meta"] = meta
    if gpu:
        print(f"[gpu] detected: {gpu}", flush=True)
    total_n = len(problems)
    cfg = LANGS[lang]

    def process_problem(i, name, p):
        """One problem end-to-end. Touches only this exercise's own dir, so
        concurrent runs (and their containers) don't collide."""
        sol_rels = p["sol_rels"]
        if len(sol_rels) == 1:
            sys_c = SYS_TMPL.format(pretty=cfg["pretty"], fence=cfg["fence"])
            usr_c = USER_TMPL.format(
                instructions=p["instructions"], fname=sol_rels[0],
                fence=cfg["fence"], stub=p["stubs"][sol_rels[0]])
        else:
            blocks = "\n\n".join(
                f"FILE: {rel}\n```{cfg['fence']}\n{p['stubs'][rel]}\n```"
                for rel in sol_rels)
            sys_c = SYS_TMPL_MULTI.format(
                pretty=cfg["pretty"], fence=cfg["fence"], n=len(sol_rels))
            usr_c = USER_TMPL_MULTI.format(
                instructions=p["instructions"], fence=cfg["fence"],
                stub_blocks=blocks)
        msgs = [{"role": "system", "content": sys_c},
                {"role": "user", "content": usr_c}]
        extra = f" +{len(sol_rels)-1} more" if len(sol_rels) > 1 else ""
        print(f"\n[{i}/{total_n}] start {name} (stub={p['sol_rel']}{extra})",
              flush=True)
        result = {"name": name, "stub": p["sol_rel"], "sol_files": sol_rels}
        try:
            enable_thinking = (args.reasoning == "enabled")
            resp, latency = chat(
                served, msgs, args.max_tokens, args.timeout,
                temperature=args.temperature,
                enable_thinking=enable_thinking,
                reasoning_effort=args.reasoning_effort,
            )
            choice = resp["choices"][0]
            content = choice["message"].get("content") or ""
            reasoning = (choice["message"].get("reasoning_content")
                         or choice["message"].get("reasoning") or "")
            usage = resp.get("usage", {})
            finish = choice.get("finish_reason")
            parsed = extract_files(content, cfg["fence"], sol_rels)
            result.update({
                "latency_s": round(latency, 1), "finish": finish,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_chars": len(reasoning),
                "code_len": sum(len(v) for v in parsed.values()),
                "files_parsed": sorted(parsed),
                "raw_content": content, "raw_reasoning": reasoning,
            })
            missing = [r for r in sol_rels if not parsed.get(r)]
            if missing:
                result.update({"ok": False,
                               "error": f"no-code-for: {','.join(missing)}"})
                print(f"[{i}/{total_n}] ✗ {name} missing {missing} | "
                      f"finish={finish} content={len(content)} "
                      f"reasoning={len(reasoning)}", flush=True)
            else:
                test_res = run_in_docker(p, lang, args.test_timeout, parsed)
                result["test"] = test_res
                result["ok"] = test_res["ok"]
                tag = "✓" if test_res["ok"] else "✗"
                print(f"[{i}/{total_n}] {tag} {name} rc={test_res['rc']} "
                      f"latency={latency:.1f}s out_tok={usage.get('completion_tokens')}",
                      flush=True)
        except urllib.error.HTTPError as e:
            err = e.read().decode()[:500] if e.fp else str(e)
            result.update({"ok": False, "error": f"http {e.code}: {err}"})
            print(f"[{i}/{total_n}] ✗ {name} HTTP {e.code}: {err}", flush=True)
        except Exception as e:
            result.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{total_n}] ✗ {name} {type(e).__name__}: {e}", flush=True)
        return result

    results_by_idx = [None] * total_n
    write_lock = threading.Lock()

    def snapshot(idx, p, result):
        with write_lock:
            results_by_idx[idx] = result
            summary["results"] = [r for r in results_by_idx if r is not None]
            summary_path.write_text(json.dumps(summary, indent=2))
            (p["dir"] / ".oneshot.results.json").write_text(
                json.dumps(result, indent=2))

    workers = max(1, min(args.concurrency, total_n))
    print(f"[run] lang={lang} concurrency={workers} image={img}", flush=True)
    if workers == 1:
        for idx, (name, p) in enumerate(problems):
            snapshot(idx, p, process_problem(idx + 1, name, p))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            # Submit all tasks upfront so workers pick up the next task
            # immediately after each one finishes (sliding window, not batches).
            # Stagger submissions slightly to avoid overwhelming the server at start.
            fut_meta = {}
            for idx, (name, p) in enumerate(problems):
                fut_meta[ex.submit(process_problem, idx + 1, name, p)] = (idx, p)
                if idx < len(problems) - 1:
                    time.sleep(0.15)

            batch_count = 0
            for fut in as_completed(fut_meta):
                idx, p = fut_meta[fut]
                snapshot(idx, p, fut.result())
                batch_count += 1
                if batch_count % workers == 0:
                    print(f"[run] Batch {batch_count // workers} complete", flush=True)

    passed = sum(1 for r in summary["results"] if r.get("ok"))
    total = len(summary["results"])
    summary["passed"] = passed
    summary["total"] = total
    summary["pass_rate"] = round(100 * passed / total, 1) if total else 0
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {args.alias} {lang}: {passed}/{total} = "
          f"{summary['pass_rate']}%  -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()

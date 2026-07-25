#!/usr/bin/env python3
"""Single-stream decode tok/s on coding prompts against an OpenAI-compatible vLLM.
Streams each prompt, measures decode throughput (tokens after first / (t_last - t_first)),
and TTFT. Reports per-prompt and mean. Usage:
  python3 dflash_speed.py --model qwen36-27b-dflash --max-tokens 4096
"""
import argparse, json, time, urllib.request, statistics as st

PROMPTS = [
    "Write a complete Python implementation of an LRU cache with O(1) get/put using a doubly linked list and dict. Include docstrings and a few unit tests.",
    "Implement a thread-safe bounded blocking queue in Go with Put/Get and a Close method. Explain the synchronization, then give the full code.",
    "Write a Rust function that parses a simple arithmetic expression (with + - * / and parentheses) into an AST and evaluates it. Provide the enum, parser, and tests.",
    "In C++, implement a templated fixed-size ring buffer with push/pop/full/empty. Show the header and a small main() demonstrating it.",
    "Write a TypeScript debounce<T> higher-order function with leading/trailing options, fully typed, plus jest tests covering both modes.",
]

def run_one(base, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer dummy"})
    t0 = time.perf_counter(); t_first = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    n += 1
    t_end = time.perf_counter()
    ctoks = usage.get("completion_tokens") if usage else n
    decode_s = (t_end - t_first) if t_first else 0
    ttft = (t_first - t0) if t_first else 0
    tps = (ctoks - 1) / decode_s if decode_s > 0 else 0  # decode-only tok/s
    return {"completion_tokens": ctoks, "ttft": ttft, "decode_s": decode_s, "tok_s": tps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    rows = []
    for i, p in enumerate(PROMPTS):
        r = run_one(a.base, a.model, p, a.max_tokens)
        rows.append(r)
        print(f"[{i+1}/{len(PROMPTS)}] ctoks={r['completion_tokens']:5d} "
              f"ttft={r['ttft']:.2f}s decode={r['decode_s']:.1f}s "
              f"=> {r['tok_s']:.1f} tok/s", flush=True)
    tps = [r["tok_s"] for r in rows]
    print(f"\nMODEL={a.model}  prompts={len(rows)}")
    print(f"  mean decode tok/s : {st.mean(tps):.1f}")
    print(f"  median            : {st.median(tps):.1f}")
    print(f"  min / max         : {min(tps):.1f} / {max(tps):.1f}")
    print(f"  mean ctoks        : {st.mean(r['completion_tokens'] for r in rows):.0f}")

if __name__ == "__main__":
    main()

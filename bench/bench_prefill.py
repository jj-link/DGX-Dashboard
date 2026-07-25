#!/usr/bin/env python3
"""Long-prompt prefill benchmark for an OpenAI-compatible endpoint."""
import argparse
import json
import random
import statistics
import sys
import time
import urllib.request

LOREM = (
    "In the field of large language model inference, prefill processes the input "
    "prompt and materializes attention state before decode begins. Chunked prefill "
    "splits long prompts into bounded token batches, which can improve scheduling "
    "but may reduce throughput when chunks are too small for the hardware. "
)


def post_json(url, body, timeout=60):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def count_tokens(base, model, text):
    data = post_json(f"{base}/tokenize", {"model": model, "prompt": text}, timeout=30)
    return data.get("count") or len(data.get("tokens", []))


def build_prompt(base, model, target_tokens, seed):
    random.seed(seed)
    nonce = f"Run nonce {random.randint(0, 1 << 31)}-{seed}.\n\n"
    task = "\n\nSummarize the reference material in one concise paragraph."
    n_chunks = max(100, target_tokens // 8)
    while True:
        text = nonce + (LOREM * n_chunks) + task
        toks = count_tokens(base, model, text)
        if toks >= target_tokens:
            while toks > target_tokens + 200 and n_chunks > 100:
                n_chunks -= 50
                text = nonce + (LOREM * n_chunks) + task
                toks = count_tokens(base, model, text)
            return text, toks
        n_chunks = int(n_chunks * (target_tokens / max(toks, 1)) * 1.05) + 100


def first_token(base, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    usage = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                return time.perf_counter() - t0, usage
    return time.perf_counter() - t0, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--prompt-tokens", type=int, default=50000)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    for i in range(args.warmup):
        try:
            ttft, _ = first_token(args.base, args.model, "Say hello.", 8)
            print(f"[warmup {i + 1}] ttft={ttft:.3f}s", flush=True)
        except Exception as exc:
            print(f"[warmup {i + 1}] FAILED: {exc}", flush=True)
            sys.exit(1)

    ttfts = []
    prompt_counts = []
    for i in range(args.runs):
        prompt, prompt_tok_actual = build_prompt(
            args.base, args.model, args.prompt_tokens, seed=i * 101 + 13
        )
        ttft, usage = first_token(args.base, args.model, prompt, args.max_tokens)
        prompt_tok = (usage or {}).get("prompt_tokens") or prompt_tok_actual
        ttfts.append(ttft)
        prompt_counts.append(prompt_tok)
        prefill_tps = prompt_tok / ttft if ttft > 0 else 0
        print(
            f"[run {i + 1}] prompt_tok={prompt_tok} ttft={ttft:.3f}s "
            f"prefill={prefill_tps:.1f} tok/s",
            flush=True,
        )

    if ttfts:
        mean_ttft = statistics.mean(ttfts)
        mean_prompt = statistics.mean(prompt_counts)
        mean_prefill = mean_prompt / mean_ttft if mean_ttft > 0 else 0
        print()
        print(
            f"==> prefill: prompt_tokens={mean_prompt:.0f} "
            f"ttft_avg={mean_ttft:.3f}s tok/s={mean_prefill:.1f} n={len(ttfts)}"
        )


if __name__ == "__main__":
    main()

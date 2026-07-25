#!/usr/bin/env python3
"""Honest 50k-context decode benchmark.

- Sends a prompt that *tokenizes* to ~50k tokens (uses /v1/tokenize to verify).
- Uses unique per-run nonce so prefix caching can't shortcut prefill.
- Generates 1024 decode tokens so we measure sustained decode, not a 256-tok blip.
- Reports per-run and aggregate decode tok/s; pass/fail vs target.
"""
import argparse, json, time, sys, urllib.request, random, statistics

LOREM = (
    "In the field of large language model inference, the dominant cost at long context "
    "is the memory bandwidth required to fetch the key-value cache during each decode "
    "step. Group-query attention reduces this cost by sharing KV heads across query "
    "heads, and hybrid linear-attention layers further amortize the per-token state. "
    "When evaluating throughput, one must distinguish prefill from decode and report "
    "tokens per second relative to the regime that bottlenecks the workload. "
)

def post_json(url, body, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

def count_tokens(base, model, text):
    r = post_json(f"{base}/tokenize", {"model": model, "prompt": text}, timeout=30)
    return r.get("count") or len(r.get("tokens", []))

def build_prompt(base, model, target_tokens, seed):
    random.seed(seed)
    nonce = f"Run nonce {random.randint(0, 1<<31)}-{seed}. Below is reference text:\n\n"
    task = ("\n\nAfter reading the above, write a detailed 800-word essay describing "
            "the role of KV cache memory bandwidth in long-context LLM decode, "
            "mentioning grouped-query attention, FlashAttention, and speculative "
            "decoding. Be specific.")
    # Start small, grow until we exceed target_tokens.
    n_chunks = 6500
    while True:
        body = LOREM * n_chunks
        text = nonce + body + task
        toks = count_tokens(base, model, text)
        if toks >= target_tokens:
            # Trim back by removing chunks until we're slightly above target
            while toks > target_tokens + 200 and n_chunks > 100:
                n_chunks -= 50
                body = LOREM * n_chunks
                text = nonce + body + task
                toks = count_tokens(base, model, text)
            return text, toks
        n_chunks = int(n_chunks * (target_tokens / max(toks, 1)) * 1.05) + 100

def post_stream(base, model, prompt, max_tokens, temperature=0.0):
    url = f"{base}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n_chunks = 0
    usage = None
    last = t0
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
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if content or reasoning:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    n_chunks += 1
                    last = now
    t_end = time.perf_counter()
    return {"ttft": ttft, "total": t_end - t0,
            "n_chunks": n_chunks, "usage": usage, "last": last}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--prompt-tokens", type=int, default=50000)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--target", type=float, default=120.0)
    args = ap.parse_args()

    # Warmup (short, just to wake the engine cache)
    for i in range(args.warmup):
        try:
            r = post_stream(args.base, args.model, "Say hello.", 16)
            print(f"[warmup {i+1}] ok ttft={r['ttft']:.3f}s", flush=True)
        except Exception as e:
            print(f"[warmup {i+1}] FAILED: {e}"); sys.exit(1)

    print(f"[build] target prompt tokens = {args.prompt_tokens}", flush=True)
    decode_tps_list = []
    actual_tok_counts = []
    for i in range(args.runs):
        prompt, prompt_tok_actual = build_prompt(args.base, args.model,
                                                 args.prompt_tokens, seed=i*101 + 7)
        r = post_stream(args.base, args.model, prompt, args.max_tokens)
        u = r["usage"] or {}
        out_tok = u.get("completion_tokens") or r["n_chunks"]
        prompt_tok_reported = u.get("prompt_tokens")
        decode_time = r["total"] - (r["ttft"] or 0)
        decode_tps = out_tok / decode_time if decode_time > 0 else 0
        decode_tps_list.append(decode_tps)
        actual_tok_counts.append(prompt_tok_reported or prompt_tok_actual)
        print(f"[run {i+1}] prompt_tok={prompt_tok_reported} "
              f"out_tok={out_tok} ttft={r['ttft']:.2f}s total={r['total']:.2f}s  "
              f"decode_time={decode_time:.2f}s  decode={decode_tps:.1f} tok/s",
              flush=True)

    if decode_tps_list:
        avg = statistics.mean(decode_tps_list)
        med = statistics.median(decode_tps_list)
        stdev = statistics.stdev(decode_tps_list) if len(decode_tps_list) > 1 else 0.0
        best = max(decode_tps_list)
        worst = min(decode_tps_list)
        prompt_mean = statistics.mean(actual_tok_counts)
        print()
        print(f"==> prompt_tokens (mean actual) = {prompt_mean:.0f}")
        print(f"==> decode tok/s: avg={avg:.1f}  median={med:.1f}  "
              f"stdev={stdev:.1f}  range=[{worst:.1f}, {best:.1f}]  n={len(decode_tps_list)}")
        verdict_avg = "MET" if avg >= args.target else "below"
        verdict_med = "MET" if med >= args.target else "below"
        print(f"==> target={args.target:.0f} tok/s -> avg {verdict_avg} "
              f"({avg/args.target*100-100:+.1f}%), median {verdict_med} "
              f"({med/args.target*100-100:+.1f}%)")

if __name__ == "__main__":
    main()

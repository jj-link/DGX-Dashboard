#!/usr/bin/env python3
"""Single-stream decode throughput benchmark for an OpenAI-compatible endpoint."""
import argparse, json, time, sys, urllib.request

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
    with urllib.request.urlopen(req, timeout=600) as resp:
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
    return {"ttft": ttft, "total": t_end - t0, "gen_time": (last - t0) - (ttft or 0),
            "n_chunks": n_chunks, "usage": usage}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    prompt = args.prompt or (
        "Write a complete, well-structured implementation of the classic Snake game "
        "in Python using pygame. Include a scoreboard, increasing difficulty, a pause "
        "feature, and clear comments. Then explain the design choices you made in detail."
    )

    for i in range(args.warmup):
        try:
            r = post_stream(args.base, args.model, "Say hello.", 16)
            print(f"[warmup {i+1}] ok ttft={r['ttft']:.3f}s")
        except Exception as e:
            print(f"[warmup {i+1}] FAILED: {e}"); sys.exit(1)

    decode_tps_list = []
    for i in range(args.runs):
        r = post_stream(args.base, args.model, prompt, args.max_tokens)
        u = r["usage"] or {}
        out_tok = u.get("completion_tokens") or r["n_chunks"]
        prompt_tok = u.get("prompt_tokens")
        gen_time = r["total"] - (r["ttft"] or 0)
        decode_tps = out_tok / gen_time if gen_time > 0 else 0
        overall_tps = out_tok / r["total"]
        decode_tps_list.append(decode_tps)
        print(f"[run {i+1}] prompt_tok={prompt_tok} out_tok={out_tok} "
              f"ttft={r['ttft']:.3f}s total={r['total']:.2f}s  "
              f"decode={decode_tps:.1f} tok/s  overall={overall_tps:.1f} tok/s")

    if decode_tps_list:
        avg = sum(decode_tps_list) / len(decode_tps_list)
        best = max(decode_tps_list)
        print(f"\n==> decode tok/s: avg={avg:.1f}  best={best:.1f}  (over {len(decode_tps_list)} runs)")

if __name__ == "__main__":
    main()

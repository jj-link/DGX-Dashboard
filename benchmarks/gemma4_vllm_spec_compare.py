#!/usr/bin/env python3
import argparse, json, time, urllib.request

PROMPTS = [
    "Write a concise explanation of speculative decoding in one paragraph.",
    "Give five practical tips for debugging CUDA out-of-memory errors in LLM serving.",
    "Summarize why Blackwell NVFP4 can improve local inference for a 31B model.",
]

def post_json(url, payload, timeout=300):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    elapsed = time.time() - t0
    return elapsed, json.loads(body)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    total_completion = 0
    total_time = 0.0
    for i, prompt in enumerate(PROMPTS, 1):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        elapsed, res = post_json(args.base + "/v1/chat/completions", payload)
        usage = res.get("usage", {})
        comp = usage.get("completion_tokens") or 0
        prompt_tokens = usage.get("prompt_tokens") or 0
        tps = comp / elapsed if elapsed > 0 else 0
        content = res["choices"][0]["message"].get("content", "")
        row = {
            "i": i,
            "elapsed_sec": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp,
            "total_tokens": usage.get("total_tokens"),
            "completion_tps": tps,
            "finish_reason": res["choices"][0].get("finish_reason"),
            "preview": content[:160],
        }
        rows.append(row)
        total_completion += comp
        total_time += elapsed
        print(json.dumps(row, ensure_ascii=False))

    summary = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "requests": len(rows),
        "total_elapsed_sec": total_time,
        "total_completion_tokens": total_completion,
        "aggregate_completion_tps": total_completion / total_time if total_time > 0 else 0,
        "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY", json.dumps({k: summary[k] for k in ["requests", "total_elapsed_sec", "total_completion_tokens", "aggregate_completion_tps"]}))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Long-context decode tok/s probe (simple, on purpose).

Builds a prompt of ~target context length from a corpus, generates a fixed
number of tokens at temp 0.6, and reports decode-only tok/s + the server's
actual prompt_tokens. Draft acceptance is read separately from the server's
SpecDecoding log lines (this client only times throughput).

  python3 dflash_longctx.py --model Qwen3.6-27B --ctx 64000
"""
import argparse, json, time, urllib.request

def run(base, model, prompt, gen, temp):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen, "temperature": temp, "top_p": 0.95,
        "stream": True, "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    t0 = time.perf_counter(); tf = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: o = json.loads(d)
            except Exception: continue
            if o.get("usage"): usage = o["usage"]
            for ch in o.get("choices", []):
                de = ch.get("delta", {})
                if de.get("content") or de.get("reasoning_content") or de.get("reasoning"):
                    if tf is None: tf = time.perf_counter()
                    n += 1
    te = time.perf_counter()
    ct = (usage or {}).get("completion_tokens") or n
    pt = (usage or {}).get("prompt_tokens")
    dec = (te - tf) if tf else 0
    tps = (ct - 1) / dec if dec > 0 else 0
    return pt, ct, (tf - t0 if tf else 0), dec, tps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", default="benchmarks/longctx_corpus.txt")
    ap.add_argument("--ctx", type=int, required=True, help="approx target prompt tokens")
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--cpt", type=float, default=3.6, help="chars/token estimate for sizing")
    a = ap.parse_args()
    corpus = open(a.corpus, encoding="utf-8", errors="ignore").read()
    need = int(a.ctx * a.cpt)
    while len(corpus) < need:  # extend if corpus too short (adds repetition)
        corpus += corpus
    body = corpus[:need]
    prompt = ("You are a senior engineer. Below is a large codebase. Study it, then "
              "implement a NEW, self-contained module that adds a feature consistent "
              "with its style. Write complete code, no ellipses, no summaries.\n\n"
              "=== CODEBASE ===\n" + body + "\n=== END CODEBASE ===\n"
              "Now write the new module, in full:")
    pt, ct, ttft, dec, tps = run(a.base, a.model, prompt, a.gen, a.temp)
    print(f"ctx_target={a.ctx} prompt_tokens={pt} gen={ct} "
          f"ttft={ttft:.1f}s decode_s={dec:.1f} tok_s={tps:.1f}", flush=True)

if __name__ == "__main__":
    main()

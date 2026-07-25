#!/usr/bin/env python3
"""Simple streaming chat script for sglang server with stats."""
import sys
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "diffusiongemma-27B-A4B-it-NVFP4"

def get_model():
    req = urllib.request.Request(f"{BASE}/v1/models")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    if data["data"]:
        return data["data"][0]["id"]
    return MODEL

def stream_chat(messages):
    payload = json.dumps({"model": MODEL, "messages": messages, "stream": True, "max_tokens": 8192}).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
    start = time.time()
    first_token = None
    total_chars = 0
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                text = line.decode().strip()
                if not text or not text.startswith("data:"):
                    continue
                if text == "data: [DONE]":
                    break
                chunk = json.loads(text[6:])
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if first_token is None:
                        first_token = time.time()
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    total_chars += len(content)
                usage = chunk.get("usage", {})
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return

    elapsed = time.time() - start
    ttft = round(first_token - start, 3) if first_token else 0
    prompt_tok = usage.get("prompt_tokens", 0)
    completion_tok = usage.get("completion_tokens", 0) or total_chars
    tok_s = round(completion_tok / elapsed, 1) if elapsed > 0 else 0
    print()
    print("[ stats: prompt=" + str(prompt_tok) + "  completion=" + str(completion_tok) + "  ttft=" + str(ttft) + "s  total=" + str(round(elapsed,2)) + "s  " + str(tok_s) + "tok/s ]")
    print()

def main():
    try:
        model = get_model()
        print("Connected to model: " + model)
        print("Type messages. Ctrl-D to quit, Ctrl-C for new conversation.\n")
    except Exception as e:
        print("ERROR: Cannot connect to sglang server (" + str(e) + ")", file=sys.stderr)
        sys.exit(1)

    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]

    while True:
        sys.stdout.write("> ")
        sys.stdout.flush()
        try:
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        stream_chat(messages)
        messages.append({"role": "assistant", "content": ""})

if __name__ == "__main__":
    main()

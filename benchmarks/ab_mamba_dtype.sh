#!/usr/bin/env bash
# A/B: does forcing the Mamba SSM-state cache to float16 cost real quality?
# Same model, same 34 polyglot Python problems, same max_tokens — only the
# --mamba-ssm-cache-dtype differs (float16 = our default, float32 = native).
#
#   ./ab_mamba_dtype.sh                 # full 34, both arms
#   NUM_TESTS=6 ./ab_mamba_dtype.sh     # quick smoke
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
SERVE_REL="serve/vllm/serve_qwen36_27b_fp8.sh"
CNAME="vllm-serve_qwen36_27b_fp8"
SERVED="Qwen3.6-27B"
BASE="http://localhost:8000"

NUM_TESTS="${NUM_TESTS:-34}"
MAX_TOKENS="${MAX_TOKENS:-32768}"   # prior data: 27B p95 out ~14k, so this is enough
TIMEOUT="${TIMEOUT:-600}"           # per-problem request timeout
READY_MAX="${READY_MAX:-1800}"      # cold weight load + torch.compile can be minutes

OUT="${SCRIPT_DIR}/results/ab-mamba-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
exec > >(tee -a "${OUT}/ab.log") 2>&1
echo "[ab] out=$OUT  N=$NUM_TESTS  max_tokens=$MAX_TOKENS"

run_arm() {
  local dt="$1"
  echo ""
  echo "================ arm: MAMBA_DTYPE=$dt ================"
  docker rm -f "$CNAME" >/dev/null 2>&1 || true

  MAMBA_DTYPE="$dt" DETACH=1 HF_CACHE_MODE=ro \
    "${REPO}/run_vllm_docker.sh" "$SERVE_REL" >"${OUT}/launch-$dt.log" 2>&1

  local up=""
  for ((i=0; i<READY_MAX; i+=10)); do
    if curl -s -m3 "$BASE/v1/models" 2>/dev/null | grep -q "$SERVED"; then
      up=1; echo "[$dt] server up after ~${i}s"; break
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "$CNAME"; then
      echo "[$dt] CONTAINER EXITED before ready — likely OOM at fp32 SSM. Tail:"
      docker logs --tail 30 "$CNAME" 2>&1 | grep -iE 'error|memory|oom|assert' | tail -10
      break
    fi
    sleep 10
  done
  if [ -z "$up" ]; then
    echo "[$dt] SERVER DID NOT COME UP — arm skipped, recorded as FAILED"
    echo '{"arm":"'"$dt"'","status":"server_failed"}' > "${OUT}/summary-$dt.json"
    docker rm -f "$CNAME" >/dev/null 2>&1 || true
    return
  fi

  OPENAI_API_BASE="$BASE/v1" OPENAI_API_KEY="dummy" \
    python3 "${SCRIPT_DIR}/oneshot_bench.py" "ab-mamba-$dt" \
      --served-model "$SERVED" \
      --num-tests "$NUM_TESTS" \
      --max-tokens "$MAX_TOKENS" \
      --timeout "$TIMEOUT" \
      --out-dir "$OUT" \
    || echo "[$dt] oneshot_bench returned non-zero (partial results kept)"

  docker rm -f "$CNAME" >/dev/null 2>&1 || true
}

run_arm float16
run_arm float32

echo ""
echo "================ DIFF ================"
python3 - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
def load(dt):
    fs = sorted(glob.glob(os.path.join(out, f"*ab-mamba-{dt}*.json")))
    fs = [f for f in fs if "summary-" not in f] or glob.glob(os.path.join(out, f"summary-{dt}.json"))
    if not fs: return None
    return json.load(open(fs[-1]))
a, b = load("float16"), load("float32")
if not a or not b or a.get("status") or b.get("status"):
    print("One arm did not produce results:",
          "fp16=", (a or {}).get("status","ok" if a else "missing"),
          "fp32=", (b or {}).get("status","ok" if b else "missing"))
    sys.exit(0)
def idx(s): return { r["name"]: bool(r.get("ok")) for r in s["results"] }
ia, ib = idx(a), idx(b)
print(f"fp16: {a['passed']}/{a['total']} = {a['pass_rate']}%")
print(f"fp32: {b['passed']}/{b['total']} = {b['pass_rate']}%")
print(f"delta (fp32 - fp16): {b['pass_rate'] - a['pass_rate']:+.1f} pts")
flips = [(k, ia.get(k), ib.get(k)) for k in sorted(set(ia)|set(ib)) if ia.get(k)!=ib.get(k)]
if flips:
    print("\nproblems that changed outcome (fp16 -> fp32):")
    for k, x, y in flips:
        print(f"  {k:30s} {'PASS' if x else 'fail'} -> {'PASS' if y else 'fail'}")
else:
    print("\nno per-problem outcome changed — fp16 SSM is quality-neutral on this set.")
PY
echo ""
echo "[ab] done. Full artifacts in $OUT"

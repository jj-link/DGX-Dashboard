#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jjlink/inference

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

(( $# >= 3 )) || fail "internal usage: run-remote-single.sh <expected-commit> <vllm|sglang> <artifact> [engine args...]"
expected_commit="$1"
engine="$2"
artifact="$3"
shift 3

[[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || fail "expected controller commit is invalid"
[[ "$engine" == vllm || "$engine" == sglang ]] || fail "invalid engine '$engine'"
[[ "$artifact" =~ ^[a-z0-9][a-z0-9_]*$ ]] || fail "invalid artifact name '$artifact'"
[[ -d "$ROOT/.git" ]] || fail "'$ROOT' is not a Git checkout; run workstation ./sync.sh for this Spark"
git -C "$ROOT" symbolic-ref -q HEAD >/dev/null || fail "'$ROOT' is detached; run workstation ./sync.sh for this Spark"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=normal)" ]] || fail "'$ROOT' is dirty; run workstation ./sync.sh for this Spark"
actual_commit="$(git -C "$ROOT" rev-parse HEAD)"
[[ "$actual_commit" == "$expected_commit" ]] || fail "'$ROOT' is at $actual_commit, expected $expected_commit; run workstation ./sync.sh for this Spark"

command -v nvidia-smi >/dev/null 2>&1 || fail "remote single-device target must have exactly one NVIDIA GB10"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
[[ "$gpu_name" == 'NVIDIA GB10' ]] || fail "remote single-device target must have exactly one NVIDIA GB10"

base="$(realpath -e "$ROOT/serve/$engine/spark")"
candidate="$base/$artifact"
if [[ ! -d "$candidate" || ! -f "$candidate/runtime.env" || ! -f "$candidate/serve.sh" ]]; then
  fail "no spark $engine recipe named '$artifact'"
fi
package="$(realpath -e "$candidate")"
script="$(realpath -e "$candidate/serve.sh")"
[[ "$package" == "$base/"* && "$script" == "$package/"* ]] || fail "recipe '$artifact' escapes '$base'"

exec "$ROOT/runtime/spark/run_${engine}_docker.sh" "$package" "$@"

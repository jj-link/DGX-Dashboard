#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "sync.sh"
EXPECTED_ORIGIN = "git@github.com:jj-link/inference-workspace.git"
HEAD = "a" * 40

FAKE_GIT = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
host = os.environ.get("FAKE_REMOTE_HOST", "local")
log = Path(os.environ["SYNC_TEST_LOG"])
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"tool": "git", "host": host, "args": args}) + "\n")

if len(args) < 3 or args[0] != "-C":
    print("unexpected fake git invocation", file=sys.stderr)
    raise SystemExit(90)
command = args[2]
remote = host != "local"

def selected(name):
    return remote and os.environ.get(name) == host

if command == "fetch":
    if (not remote and os.environ.get("FAKE_LOCAL_FETCH_FAIL") == "1") or selected("FAKE_REMOTE_FETCH_FAIL"):
        raise SystemExit(1)
    raise SystemExit(0)
if command == "status":
    if (not remote and os.environ.get("FAKE_LOCAL_DIRTY") == "1") or selected("FAKE_REMOTE_DIRTY"):
        print(" M changed")
    raise SystemExit(0)
if command == "remote" and args[3:5] == ["get-url", "origin"]:
    if (not remote and os.environ.get("FAKE_LOCAL_WRONG_ORIGIN") == "1") or selected("FAKE_REMOTE_WRONG_ORIGIN"):
        print("git@example.invalid:wrong/repo.git")
    else:
        print(os.environ["EXPECTED_ORIGIN"])
    raise SystemExit(0)
if command == "rev-parse":
    value = args[3]
    if value == "origin/main" and not remote and os.environ.get("FAKE_LOCAL_BEHIND") == "1":
        print("b" * 40)
    else:
        print(os.environ["FAKE_HEAD"])
    raise SystemExit(0)
if command == "symbolic-ref":
    if selected("FAKE_REMOTE_DETACHED"):
        raise SystemExit(1)
    print("refs/heads/main")
    raise SystemExit(0)
if command == "merge":
    if selected("FAKE_REMOTE_DIVERGENT"):
        raise SystemExit(1)
    raise SystemExit(0)
print(f"unhandled fake git command: {args}", file=sys.stderr)
raise SystemExit(91)
'''

FAKE_SSH = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys

args = sys.argv[1:]
if len(args) < 4 or args[:2] != ["-o", "BatchMode=yes"]:
    print(f"unexpected fake ssh invocation: {args}", file=sys.stderr)
    raise SystemExit(92)
host = args[2]
log = Path(os.environ["SYNC_TEST_LOG"])
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"tool": "ssh", "host": host, "args": args[3:]}) + "\n")
if os.environ.get("FAKE_OFFLINE_HOST") == host:
    print("offline", file=sys.stderr)
    raise SystemExit(255)
environment = os.environ.copy()
environment["FAKE_REMOTE_HOST"] = host
completed = subprocess.run(
    args[3:],
    input=sys.stdin.buffer.read(),
    env=environment,
    stdout=sys.stdout.buffer,
    stderr=sys.stderr.buffer,
)
raise SystemExit(completed.returncode)
'''


def executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def invoke(arguments: list[str], **overrides: str) -> tuple[subprocess.CompletedProcess[str], list[dict[str, object]]]:
    with tempfile.TemporaryDirectory() as temporary:
        temporary_path = Path(temporary)
        fake_bin = temporary_path / "bin"
        fake_bin.mkdir()
        executable(fake_bin / "git", FAKE_GIT)
        executable(fake_bin / "ssh", FAKE_SSH)
        log = temporary_path / "calls.jsonl"
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "SYNC_TEST_LOG": str(log),
                "EXPECTED_ORIGIN": EXPECTED_ORIGIN,
                "FAKE_HEAD": HEAD,
            }
        )
        environment.update(overrides)
        completed = subprocess.run(
            [str(SYNC), *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        return completed, calls


def ssh_hosts(calls: list[dict[str, object]]) -> list[str]:
    return [str(call["host"]) for call in calls if call["tool"] == "ssh"]


def assert_safe_git_calls(calls: list[dict[str, object]]) -> None:
    forbidden = {"reset", "clean", "stash", "push", "commit"}
    for call in calls:
        if call["tool"] != "git":
            continue
        arguments = set(str(item) for item in call["args"])
        assert not arguments & forbidden, call


def main() -> int:
    for arguments in ([], ["unknown"], ["all", "spark1-ts"]):
        completed, calls = invoke(arguments)
        assert completed.returncode == 2, (arguments, completed)
        assert ssh_hosts(calls) == []

    completed, calls = invoke(["all"])
    assert completed.returncode == 0, completed.stderr
    assert ssh_hosts(calls) == ["spark1-ts", "spark2-ts", "spark3-ts"]
    assert all(f"{host}: ok {HEAD}" in completed.stdout for host in ("spark1-ts", "spark2-ts", "spark3-ts"))
    assert_safe_git_calls(calls)

    completed, calls = invoke(["spark3-ts", "spark1-ts", "spark3-ts", "spark2-ts"])
    assert completed.returncode == 0, completed.stderr
    assert ssh_hosts(calls) == ["spark1-ts", "spark2-ts", "spark3-ts"]

    completed, calls = invoke(["all"], FAKE_OFFLINE_HOST="spark2-ts")
    assert completed.returncode == 1
    assert ssh_hosts(calls) == ["spark1-ts", "spark2-ts", "spark3-ts"]
    assert "spark2-ts: failed" in completed.stderr
    assert "spark3-ts: ok" in completed.stdout

    for variable in ("FAKE_LOCAL_DIRTY", "FAKE_LOCAL_FETCH_FAIL", "FAKE_LOCAL_BEHIND", "FAKE_LOCAL_WRONG_ORIGIN"):
        completed, calls = invoke(["spark1-ts"], **{variable: "1"})
        assert completed.returncode == 1, (variable, completed)
        assert ssh_hosts(calls) == [], variable

    remote_failures = {
        "FAKE_REMOTE_DIRTY": "dirty",
        "FAKE_REMOTE_FETCH_FAIL": "fetch",
        "FAKE_REMOTE_DETACHED": "detached",
        "FAKE_REMOTE_WRONG_ORIGIN": "origin",
        "FAKE_REMOTE_DIVERGENT": "fast-forward",
    }
    for variable, diagnostic in remote_failures.items():
        completed, calls = invoke(["spark2-ts"], **{variable: "spark2-ts"})
        assert completed.returncode == 1, (variable, completed)
        assert ssh_hosts(calls) == ["spark2-ts"]
        assert diagnostic in completed.stderr, (variable, completed.stderr)
        assert_safe_git_calls(calls)

    print("sync contracts: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

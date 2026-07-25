#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SYNC = ROOT / "sync.sh"
EXPECTED_ORIGIN = "git@github.com:jj-link/DGX-Dashboard.git"
BRANCH = "unified-control-plane"
HEAD = "a" * 40

FAKE_GIT = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
host = os.environ.get("FAKE_REMOTE_HOST", "local")
remote = host != "local"
log = Path(os.environ["SYNC_TEST_LOG"])
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"tool": "git", "host": host, "args": args}) + "\n")

def selected(name):
    return remote and os.environ.get(name) == host

if args[:2] == ["check-ref-format", "--branch"]:
    raise SystemExit(0)

if args and args[0] == "clone":
    if selected("FAKE_REMOTE_CLONE_FAIL"):
        raise SystemExit(1)
    destination = Path(args[-1])
    destination.mkdir(parents=True)
    (destination / ".git").mkdir()
    raise SystemExit(0)

if len(args) < 3 or args[0] != "-C":
    print("unexpected fake git invocation", file=sys.stderr)
    raise SystemExit(90)

repo = Path(args[1])
command = args[2]

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
    if value == "--is-inside-work-tree":
        if selected("FAKE_REMOTE_LEGACY") and not str(repo).endswith(".clone-in-progress"):
            raise SystemExit(1)
        print("true")
    elif value == f"origin/{os.environ['FAKE_BRANCH']}" and not remote and os.environ.get("FAKE_LOCAL_BEHIND") == "1":
        print("b" * 40)
    elif value == "HEAD" and selected("FAKE_REMOTE_HEAD_MISMATCH"):
        print("b" * 40)
    else:
        print(os.environ["FAKE_HEAD"])
    raise SystemExit(0)
if command == "symbolic-ref":
    if (not remote and os.environ.get("FAKE_LOCAL_DETACHED") == "1") or selected("FAKE_REMOTE_DETACHED"):
        raise SystemExit(1)
    if selected("FAKE_REMOTE_WRONG_BRANCH"):
        print("other-branch")
    else:
        print(os.environ["FAKE_BRANCH"])
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
import shutil
import subprocess
import sys

args = sys.argv[1:]
try:
    command_index = args.index("bash")
except ValueError:
    print(f"unexpected fake ssh invocation: {args}", file=sys.stderr)
    raise SystemExit(92)
if args[:2] != ["-o", "BatchMode=yes"] or args[command_index:command_index + 3] != ["bash", "-s", "--"]:
    print(f"unexpected fake ssh invocation: {args}", file=sys.stderr)
    raise SystemExit(92)
host = args[command_index - 1]
log = Path(os.environ["SYNC_TEST_LOG"])
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"tool": "ssh", "host": host, "args": args[command_index:]}) + "\n")
if os.environ.get("FAKE_OFFLINE_HOST") == host:
    print("offline", file=sys.stderr)
    raise SystemExit(255)

repo = Path(os.environ["SYNC_TEST_REMOTE_ROOT"]) / host / "dgx-dashboard"
backup = Path(f"{repo}.legacy-pre-unified-20260725")
clone = Path(f"{repo}.clone-in-progress")
repo.parent.mkdir(parents=True, exist_ok=True)
if os.environ.get("FAKE_REMOTE_MISSING") == host:
    shutil.rmtree(repo, ignore_errors=True)
else:
    repo.mkdir(exist_ok=True)
if os.environ.get("FAKE_REMOTE_LEGACY") == host:
    (repo / "legacy-marker").write_text("preserved", encoding="utf-8")
if os.environ.get("FAKE_REMOTE_BACKUP_EXISTS") == host:
    backup.mkdir(exist_ok=True)
    (backup / "existing-marker").write_text("do not overwrite", encoding="utf-8")

remote_args = list(args[command_index:])
remote_args[3] = str(repo)
environment = os.environ.copy()
environment["FAKE_REMOTE_HOST"] = host
completed = subprocess.run(
    remote_args,
    input=sys.stdin.buffer.read(),
    env=environment,
    stdout=sys.stdout.buffer,
    stderr=sys.stderr.buffer,
)
with log.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "tool": "state",
        "host": host,
        "repo_exists": repo.exists(),
        "repo_legacy_marker": (repo / "legacy-marker").exists(),
        "backup_exists": backup.exists(),
        "backup_legacy_marker": (backup / "legacy-marker").exists(),
        "backup_existing_marker": (backup / "existing-marker").exists(),
        "clone_exists": clone.exists(),
    }) + "\n")
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
        remote_root = temporary_path / "remotes"
        environment.update(
            {
                "PATH": f"{fake_bin}:{environment['PATH']}",
                "SYNC_TEST_LOG": str(log),
                "SYNC_TEST_REMOTE_ROOT": str(remote_root),
                "EXPECTED_ORIGIN": EXPECTED_ORIGIN,
                "FAKE_BRANCH": BRANCH,
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

def host_state(calls: list[dict[str, object]], host: str) -> dict[str, object]:
    return next(call for call in reversed(calls) if call["tool"] == "state" and call["host"] == host)


def host_git_calls(calls: list[dict[str, object]], host: str) -> list[dict[str, object]]:
    return [call for call in calls if call["tool"] == "git" and call["host"] == host]


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

    for variable in (
        "FAKE_LOCAL_DIRTY",
        "FAKE_LOCAL_FETCH_FAIL",
        "FAKE_LOCAL_BEHIND",
        "FAKE_LOCAL_WRONG_ORIGIN",
        "FAKE_LOCAL_DETACHED",
    ):
        completed, calls = invoke(["spark1-ts"], **{variable: "1"})
        assert completed.returncode == 1, (variable, completed)
        assert ssh_hosts(calls) == [], variable

    remote_failures = {
        "FAKE_REMOTE_DIRTY": "dirty",
        "FAKE_REMOTE_FETCH_FAIL": "fetch",
        "FAKE_REMOTE_DETACHED": "detached",
        "FAKE_REMOTE_WRONG_BRANCH": "branch",
        "FAKE_REMOTE_WRONG_ORIGIN": "origin",
        "FAKE_REMOTE_DIVERGENT": "fast-forward",
        "FAKE_REMOTE_HEAD_MISMATCH": "expected",
    }
    for variable, diagnostic in remote_failures.items():
        completed, calls = invoke(["spark2-ts"], **{variable: "spark2-ts"})
        assert completed.returncode == 1, (variable, completed)
        assert ssh_hosts(calls) == ["spark2-ts"]
        assert diagnostic in completed.stderr, (variable, completed.stderr)
        assert_safe_git_calls(calls)

    completed, calls = invoke(["spark2-ts"], FAKE_REMOTE_MISSING="spark2-ts")
    assert completed.returncode == 0, completed.stderr
    assert any(call["args"][0] == "clone" for call in host_git_calls(calls, "spark2-ts"))
    assert host_state(calls, "spark2-ts")["repo_exists"]
    assert not host_state(calls, "spark2-ts")["backup_exists"]

    completed, calls = invoke(["spark1-ts"], FAKE_REMOTE_LEGACY="spark1-ts")
    assert completed.returncode == 0, completed.stderr
    state = host_state(calls, "spark1-ts")
    assert state["repo_exists"] and state["backup_exists"]
    assert not state["repo_legacy_marker"] and state["backup_legacy_marker"]

    completed, calls = invoke(["spark2-ts"], FAKE_REMOTE_LEGACY="spark2-ts")
    assert completed.returncode == 1
    assert "not a Git checkout" in completed.stderr
    state = host_state(calls, "spark2-ts")
    assert state["repo_legacy_marker"] and not state["backup_exists"]
    assert not any(call["args"][0] == "clone" for call in host_git_calls(calls, "spark2-ts"))

    completed, calls = invoke(
        ["spark1-ts"],
        FAKE_REMOTE_LEGACY="spark1-ts",
        FAKE_REMOTE_BACKUP_EXISTS="spark1-ts",
    )
    assert completed.returncode == 1
    assert "refusing to overwrite" in completed.stderr
    state = host_state(calls, "spark1-ts")
    assert state["repo_legacy_marker"] and state["backup_existing_marker"]

    completed, calls = invoke(
        ["spark1-ts"],
        FAKE_REMOTE_LEGACY="spark1-ts",
        FAKE_REMOTE_CLONE_FAIL="spark1-ts",
    )
    assert completed.returncode == 1
    assert "clone failed" in completed.stderr
    state = host_state(calls, "spark1-ts")
    assert state["repo_legacy_marker"] and not state["backup_exists"] and not state["clone_exists"]
    assert_safe_git_calls(calls)

    print("sync contracts: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

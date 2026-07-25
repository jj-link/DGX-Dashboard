#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONT = ROOT / "serve.sh"
CONTROLLER = ROOT / "serve" / "cluster" / "sglang" / "unsloth_qwen36_27b_nvfp4_dflash_tp2" / "cluster.sh"
NODE = ROOT / "runtime" / "cluster" / "run-node.sh"
COMMIT = "a" * 40


def run(command: list[str], environment: dict[str, str], timeout: int = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=timeout)


def executable(path: pathlib.Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def base_environment(home: pathlib.Path, fake_bin: pathlib.Path) -> dict[str, str]:
    return {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(home),
        "LC_ALL": "C",
        "EXPECTED_COMMIT": COMMIT,
    }


def install_fake_git(fake_bin: pathlib.Path) -> None:
    executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
case "$*" in
  *"symbolic-ref -q HEAD"*) printf '%s\n' refs/heads/main ;;
  *"status --porcelain"*) ;;
  *"rev-parse HEAD"*) printf '%s\n' aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ;;
  *) exit 1 ;;
esac
""",
    )


def install_controller_fakes(fake_bin: pathlib.Path) -> None:
    install_fake_git(fake_bin)
    executable(
        fake_bin / "ssh",
        """#!/usr/bin/env python3
import json, os, pathlib, re, sys
arguments = sys.argv[1:]
remote = arguments[-1]
host = next((value for value in arguments if value in ("spark2-ts", "spark3-ts")), "unknown")
match = re.search(r"run-node\\.sh'?\\s+(preflight|start|wait-rank|status|logs|verify|stop|port-clear)\\s+(vllm|sglang)\\s+([a-z0-9_]+)\\s+([01])", remote)
action, engine, artifact, rank = match.groups() if match else ("unknown", "", "", "")
record = {"host": host, "action": action, "engine": engine, "artifact": artifact, "rank": rank, "remote": remote}
with pathlib.Path(os.environ["SSH_CAPTURE"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\\n")
scenario = os.environ.get("SCENARIO", "success")
if action == "preflight":
    print(f"HOST={host}")
    print(f"RANK={rank}")
    if rank == "0": print("API_HOST=127.0.0.1")
    if scenario == "preflight-head-fail" and rank == "0": sys.exit(19)
if action == "start":
    if scenario == "worker-start-fail" and rank == "1": sys.exit(20)
    if scenario == "head-start-fail" and rank == "0": sys.exit(21)
if action == "wait-rank" and scenario == "worker-wait-fail" and rank == "1": sys.exit(22)
sys.exit(0)
""",
    )
    executable(fake_bin / "curl", "#!/usr/bin/env bash\nexit 0\n")
    executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\nprintf called >\"${NVIDIA_CAPTURE:?}\"\nexit 99\n",
    )


def records(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def action_pairs(entries: list[dict[str, str]]) -> list[tuple[str, str]]:
    return [(entry["action"], entry["rank"]) for entry in entries]


def test_cluster_front_door() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        install_controller_fakes(fake_bin)
        capture = temp / "ssh.jsonl"
        nvidia_capture = temp / "nvidia-called"
        environment = base_environment(temp / "home", fake_bin)
        environment.update({"SSH_CAPTURE": str(capture), "NVIDIA_CAPTURE": str(nvidia_capture)})

        completed = run([
            str(FRONT), "cluster", "sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2", "logs", "17"
        ], environment)
        assert completed.returncode == 0, completed.stderr
        assert "target=cluster engine=sglang artifact=unsloth_qwen36_27b_nvfp4_dflash_tp2 action=logs" in completed.stderr
        entries = records(capture)
        assert action_pairs(entries) == [("logs", "0"), ("logs", "1")]
        assert all("LOG_LINES=17" in entry["remote"] for entry in entries)
        assert not nvidia_capture.exists()

        capture.unlink()
        environment["PREFLIGHT_ONLY"] = "1"
        old_image = "sha256:" + "2" * 64
        environment.update({
            "PREFLIGHT_REPLACE_CONTAINER": "old-production",
            "PREFLIGHT_REPLACE_IMAGE_ID": old_image,
            "PREFLIGHT_REPLACE_NETWORK_MODE": "host",
        })
        completed = run([
            str(FRONT), "cluster", "sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2"
        ], environment)
        assert completed.returncode == 0, completed.stderr
        assert "action=start" in completed.stderr
        assert "preflight complete" in completed.stdout
        assert action_pairs(records(capture)) == [("preflight", "0"), ("preflight", "1")]
        assert all("PREFLIGHT_REPLACE_CONTAINER=old-production" in entry["remote"] for entry in records(capture))
        assert all(f"PREFLIGHT_REPLACE_IMAGE_ID={old_image}" in entry["remote"] for entry in records(capture))
        assert all("PREFLIGHT_REPLACE_NETWORK_MODE=host" in entry["remote"] for entry in records(capture))
        assert not nvidia_capture.exists()

        for artifact in ("../escape", "/tmp/escape", "bad.name"):
            rejected = run([str(FRONT), "cluster", "sglang", artifact], environment)
            assert rejected.returncode == 1
            assert "invalid artifact name" in rejected.stderr
        rejected = run([str(FRONT), "cluster", "sglang", "does_not_exist"], environment)
        assert rejected.returncode == 1
        assert "no cluster sglang recipe named 'does_not_exist'" in rejected.stderr
        assert "unsloth_qwen36_27b_nvfp4_dflash_tp2" in rejected.stderr
        rejected = run([
            str(FRONT), "cluster", "sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2", "destroy"
        ], environment)
        assert rejected.returncode == 1
        assert "invalid cluster action 'destroy'" in rejected.stderr


def run_controller_scenario(temp: pathlib.Path, scenario: str, preflight_only: bool = False) -> tuple[subprocess.CompletedProcess[str], list[dict[str, str]]]:
    fake_bin = temp / "bin"
    fake_bin.mkdir()
    install_controller_fakes(fake_bin)
    capture = temp / "ssh.jsonl"
    environment = base_environment(temp / "home", fake_bin)
    environment.update({
        "SSH_CAPTURE": str(capture),
        "NVIDIA_CAPTURE": str(temp / "nvidia-called"),
        "SCENARIO": scenario,
    })
    if preflight_only:
        environment["PREFLIGHT_ONLY"] = "1"
    completed = run([str(CONTROLLER), "start"], environment, timeout=120)
    return completed, records(capture)


def test_transactional_failures() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        completed, entries = run_controller_scenario(pathlib.Path(temporary), "preflight-head-fail", True)
        assert completed.returncode != 0
        pairs = action_pairs(entries)
        assert ("preflight", "0") in pairs and ("preflight", "1") in pairs
        assert not any(action == "start" for action, _ in pairs)
        assert not any(action == "stop" for action, _ in pairs)

    with tempfile.TemporaryDirectory() as temporary:
        completed, entries = run_controller_scenario(pathlib.Path(temporary), "worker-wait-fail")
        assert completed.returncode != 0
        pairs = action_pairs(entries)
        assert ("start", "1") in pairs and ("wait-rank", "1") in pairs
        assert ("start", "0") not in pairs
        assert ("stop", "1") in pairs and ("stop", "0") not in pairs
        assert ("port-clear", "0") in pairs and ("port-clear", "1") in pairs

    with tempfile.TemporaryDirectory() as temporary:
        completed, entries = run_controller_scenario(pathlib.Path(temporary), "head-start-fail")
        assert completed.returncode != 0
        pairs = action_pairs(entries)
        assert ("start", "1") in pairs and ("start", "0") in pairs
        assert ("stop", "1") in pairs and ("stop", "0") not in pairs
        assert ("port-clear", "0") in pairs and ("port-clear", "1") in pairs

    with tempfile.TemporaryDirectory() as temporary:
        completed, entries = run_controller_scenario(pathlib.Path(temporary), "api-fail")
        assert completed.returncode != 0
        pairs = action_pairs(entries)
        assert ("start", "1") in pairs and ("start", "0") in pairs
        assert ("verify", "0") in pairs and ("verify", "1") in pairs
        assert ("stop", "0") in pairs and ("stop", "1") in pairs
        assert ("port-clear", "0") in pairs and ("port-clear", "1") in pairs


def assignments(path: pathlib.Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


def make_cache(cache: pathlib.Path, repository: str, revision: str) -> None:
    (cache / "hub" / f"models--{repository.replace('/', '--')}" / "snapshots" / revision).mkdir(parents=True)


def install_node_fakes(fake_bin: pathlib.Path) -> None:
    install_fake_git(fake_bin)
    executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nprintf '%s\n' 'NVIDIA GB10'\n")
    executable(fake_bin / "tailscale", "#!/usr/bin/env bash\nprintf '%s\n' '100.64.0.8'\n")
    executable(
        fake_bin / "ip",
        """#!/usr/bin/env bash
if [[ "$*" == *"link show dev"* ]]; then
  [[ "${@: -1}" == enp1s0f1np1 ]] || exit 1
  printf '2: %s: <UP> mtu 9000 state UP\n' "${@: -1}"
elif [[ "$*" == *"-4 -o addr show"* ]]; then
  printf '%s\n' '2: enp1s0f1np1 inet 10.0.0.1/24 scope global' '3: enp1s0f1np1 inet 10.0.0.2/24 scope global'
else
  exit 1
fi
""",
    )
    executable(fake_bin / "ibv_devinfo", "#!/usr/bin/env bash\nexit 0\n")
    executable(
        fake_bin / "ss",
        "#!/usr/bin/env bash\n[[ \"${FAKE_PORT_IN_USE:-0}\" == 1 ]] && printf '%s\n' 'LISTEN 0 128 0.0.0.0:8888 0.0.0.0:*'\nexit 0\n",
    )
    executable(
        fake_bin / "docker",
        """#!/usr/bin/env python3
import json, os, pathlib, sys
arguments = sys.argv[1:]
with pathlib.Path(os.environ["DOCKER_CAPTURE"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\\n")
if arguments[:2] == ["container", "inspect"]:
    if arguments[2] == "old-production":
        print(json.dumps([{
            "State": {"Running": True},
            "Image": os.environ["FAKE_OLD_IMAGE_ID"],
            "HostConfig": {"NetworkMode": "host"},
        }]))
        sys.exit(0)
    sys.exit(1)
if arguments and arguments[0] == "run":
    print("new-container-id")
sys.exit(0)
""",
    )


def docker_calls(path: pathlib.Path) -> list[list[str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_pair(arguments: list[str], option: str, value: str) -> None:
    index = arguments.index(option)
    assert arguments[index + 1] == value, (option, value, arguments[index + 1])


def test_node_preflight_and_launch() -> None:
    packages = {
        ("sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2"): ROOT / "serve" / "cluster" / "sglang" / "unsloth_qwen36_27b_nvfp4_dflash_tp2",
        ("vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2"): ROOT / "serve" / "cluster" / "vllm" / "deepseek_ai_deepseek_v4_flash_dspark_tp2",
    }
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        install_node_fakes(fake_bin)
        cache = temp / "hf"
        for package in packages.values():
            metadata = assignments(package / "runtime.env")
            make_cache(cache, metadata["MODEL"], metadata["MODEL_REVISION"])
            if metadata["DRAFTER"]:
                make_cache(cache, metadata["DRAFTER"], metadata["DRAFTER_REVISION"])
        capture = temp / "docker.jsonl"
        environment = base_environment(temp / "home", fake_bin)
        environment.update({
            "DOCKER_CAPTURE": str(capture),
            "HF_CACHE": str(cache),
            "INFERENCE_ROOT": str(ROOT),
        })

        for engine, artifact in packages:
            for rank in ("0", "1"):
                completed = run([str(NODE), "preflight", engine, artifact, rank], environment)
                assert completed.returncode == 0, (engine, rank, completed.stderr)
                assert f"RANK={rank}" in completed.stdout
                assert "MODEL_REPO_HOST=" in completed.stdout
                if rank == "0":
                    assert "API_HOST=100.64.0.8" in completed.stdout

        old_image = "sha256:" + "2" * 64
        replacement_environment = dict(environment)
        replacement_environment.update({
            "PREFLIGHT_REPLACE_CONTAINER": "old-production",
            "PREFLIGHT_REPLACE_IMAGE_ID": old_image,
            "PREFLIGHT_REPLACE_NETWORK_MODE": "host",
            "FAKE_OLD_IMAGE_ID": old_image,
            "FAKE_PORT_IN_USE": "1",
        })
        completed = run(
            [str(NODE), "preflight", "vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "0"],
            replacement_environment,
        )
        assert completed.returncode == 0, completed.stderr
        worker_replacement_environment = dict(replacement_environment)
        worker_replacement_environment.pop("FAKE_PORT_IN_USE")
        completed = run(
            [str(NODE), "preflight", "vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "1"],
            worker_replacement_environment,
        )
        assert completed.returncode == 0, completed.stderr
        mismatched_environment = dict(replacement_environment)
        mismatched_environment["PREFLIGHT_REPLACE_IMAGE_ID"] = "sha256:" + "3" * 64
        rejected = run(
            [str(NODE), "preflight", "vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "0"],
            mismatched_environment,
        )
        assert rejected.returncode == 1
        assert "replacement container 'old-production' image is" in rejected.stderr

        launch_cases = [
            ("sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2", "0"),
            ("sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2", "1"),
            ("vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "0"),
        ]
        for engine, artifact, rank in launch_cases:
            capture.write_text("", encoding="utf-8")
            completed = run([str(NODE), "start", engine, artifact, rank], environment)
            assert completed.returncode == 0, (engine, rank, completed.stderr)
            calls = docker_calls(capture)
            run_arguments = next(arguments for arguments in calls if arguments and arguments[0] == "run")
            assert "--privileged" not in run_arguments
            for required in ("--gpus", "--network", "--ipc", "--device", "--cap-drop", "--security-opt", "--read-only", "--pids-limit"):
                assert required in run_arguments, (engine, rank, required)
            assert_pair(run_arguments, "--network", "host")
            assert_pair(run_arguments, "--hostname", f"inference-{engine}-rank{rank}")
            assert_pair(run_arguments, "--device", "/dev/infiniband:/dev/infiniband")
            assert_pair(run_arguments, "--cap-drop", "ALL")
            assert_pair(run_arguments, "--security-opt", "no-new-privileges:true")
            expected_api = "100.64.0.8" if rank == "0" else "127.0.0.1"
            assert f"API_HOST={expected_api}" in run_arguments
            assert "HOME=/root/.cache" in run_arguments
            assert "WORLD_SIZE=2" in run_arguments
            assert f"NODE_RANK={rank}" in run_arguments
            assert "NCCL_NET=IB" in run_arguments
            assert "NCCL_IB_HCA=rocep1s0f1" in run_arguments
            assert "NCCL_SOCKET_IFNAME=enp1s0f1np1" in run_arguments
            assert f"{ROOT}:{ROOT}:ro" in run_arguments
            assert run_arguments[-3:] == [str(ROOT / "runtime" / "cluster" / "serve-node.sh"), engine, artifact]
            assert any(value.endswith(":ro") and "models--" in value for value in run_arguments)
            if engine == "vllm":
                assert "VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0" in run_arguments
                assert "KV_CACHE_DTYPE=fp8_ds_mla" in run_arguments
            else:
                assert "DRAFTER_PATH=" not in run_arguments
                assert any(value.startswith("DRAFTER_PATH=/models/hub/models--") for value in run_arguments)


def main() -> int:
    test_cluster_front_door()
    test_transactional_failures()
    test_node_preflight_and_launch()
    print("cluster contracts: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

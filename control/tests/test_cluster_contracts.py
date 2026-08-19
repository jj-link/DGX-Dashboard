#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
FRONT = ROOT / "serve.sh"
CONTROLLER = ROOT / "serve" / "cluster" / "sglang" / "unsloth_qwen36_27b_nvfp4_dflash_tp2" / "cluster.sh"
NODE = ROOT / "runtime" / "cluster" / "run-node.sh"
SERVE_NODE = ROOT / "runtime" / "cluster" / "serve-node.sh"
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
        fake_bin / "python3",
        """#!/usr/bin/env bash
if [[ "${1:-}" == - ]]; then
  cat >/dev/null
  [[ "${SCENARIO:-success}" != api-fail ]]
  exit
fi
exec /usr/bin/python3 "$@"
""",
    )
    executable(
        fake_bin / "ssh",
        """#!/usr/bin/python3
import json, os, pathlib, re, sys
arguments = sys.argv[1:]
remote = arguments[-1]
host = next((value for value in arguments if value in ("spark2-ts", "spark3-ts")), "unknown")
match = re.search(r"run-node\\.sh'?\\s+(preflight|start|wait-rank|status|logs|verify|stop|port-clear|api-host)\\s+(vllm|sglang)\\s+([a-z0-9_]+)\\s+([01])", remote)
action, engine, artifact, rank = match.groups() if match else ("unknown", "", "", "")
record = {"host": host, "action": action, "engine": engine, "artifact": artifact, "rank": rank, "remote": remote, "arguments": arguments}
with pathlib.Path(os.environ["SSH_CAPTURE"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\\n")
scenario = os.environ.get("SCENARIO", "success")
if action == "preflight":
    print(f"HOST={host}")
    print(f"RANK={rank}")
    if rank == "0": print("API_HOST=127.0.0.1")
    if scenario == "preflight-head-fail" and rank == "0": sys.exit(19)
if action == "api-host": print("127.0.0.1")
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
        assert all(
            any(
                entry["arguments"][index : index + 2] == ["-o", option]
                for index in range(len(entry["arguments"]) - 1)
            )
            for entry in entries
            for option in ("ConnectTimeout=20", "ConnectionAttempts=3")
        )
        assert not nvidia_capture.exists()

        capture.unlink()
        environment["PREFLIGHT_ONLY"] = "1"
        old_image = "sha256:" + "2" * 64
        environment.update({
            "PREFLIGHT_REPLACE_CONTAINER": "old-production-head,old-production-worker",
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
        assert "PREFLIGHT_REPLACE_CONTAINER=old-production-head" in records(capture)[0]["remote"]
        assert "PREFLIGHT_REPLACE_CONTAINER=old-production-worker" in records(capture)[1]["remote"]
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


def test_profiled_cluster_front_door() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        install_controller_fakes(fake_bin)
        capture = temp / "ssh.jsonl"
        environment = base_environment(temp / "home", fake_bin)
        environment.update({
            "SSH_CAPTURE": str(capture),
            "NVIDIA_CAPTURE": str(temp / "nvidia-called"),
            "PREFLIGHT_ONLY": "1",
        })
        artifact = "deepseek_ai_deepseek_v4_flash_dspark_tp2"
        expected_actions = {
            "start": [("preflight", "0"), ("preflight", "1")],
            "status": [("status", "0"), ("status", "1")],
            "logs": [("logs", "0"), ("logs", "1")],
            "verify": [("api-host", "0"), ("verify", "0"), ("verify", "1")],
            "stop": [("stop", "0"), ("stop", "1"), ("port-clear", "0"), ("port-clear", "1")],
        }

        for profile in ("quality", "balanced", "throughput"):
            for action, expected in expected_actions.items():
                capture.unlink(missing_ok=True)
                command = [str(FRONT), "cluster", "vllm", artifact, action, profile]
                if action == "logs":
                    command.append("17")
                completed = run(command, environment)
                assert completed.returncode == 0, completed.stderr
                entries = records(capture)
                assert action_pairs(entries) == expected
                assert all(f"CLUSTER_PROFILE={profile}" in entry["remote"] for entry in entries)
                if action == "logs":
                    assert all("LOG_LINES=17" in entry["remote"] for entry in entries)

        capture.unlink(missing_ok=True)
        completed = run([str(FRONT), "cluster", "vllm", artifact, "logs", "17"], environment)
        assert completed.returncode == 0, completed.stderr
        assert all("CLUSTER_PROFILE=quality" in entry["remote"] for entry in records(capture))

        capture.unlink(missing_ok=True)
        rejected = run([str(FRONT), "cluster", "vllm", artifact, "status", "unknown"], environment)
        assert rejected.returncode == 1
        assert "unknown DeepSeek cluster profile 'unknown'" in rejected.stderr
        assert records(capture) == []


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
        fake_bin / "show_gids",
        """#!/usr/bin/env bash
printf '%s\\n' \\
  'DEV PORT INDEX GID IPv4 VER DEV' \\
  'rocep1s0f1 1 3 0000:0000:0000:0000:ffff:0a00:0001 10.0.0.1 v2 enp1s0f1np1' \\
  'rocep1s0f1 1 3 0000:0000:0000:0000:ffff:0a00:0002 10.0.0.2 v2 enp1s0f1np1'
""",
    )
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
    if arguments[2] in {"old-production", os.environ.get("FAKE_TARGET_CONTAINER")}:
        print(json.dumps([{
            "State": {"Running": True},
            "Image": os.environ["FAKE_OLD_IMAGE_ID"],
            "HostConfig": {"NetworkMode": "host"},
        }]))
        sys.exit(0)
    sys.exit(1)
profile = os.environ.get("FAKE_RUNNING_PROFILE")
if arguments and arguments[0] == "inspect" and profile:
    if "-f" in arguments:
        template = arguments[arguments.index("-f") + 1]
        if ".State.Running" in template:
            print("true")
        elif ".Config.Env" in template:
            print(f"CLUSTER_PROFILE={profile}")
        sys.exit(0)
    print(json.dumps([{"State": {"Running": True}, "Config": {"Env": [f"CLUSTER_PROFILE={profile}"]}}]))
    sys.exit(0)
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
        ("vllm", "drowzeys_keys_deepseek_v4_flash_dspark_abliterated_32_32"): ROOT / "serve" / "cluster" / "vllm" / "drowzeys_keys_deepseek_v4_flash_dspark_abliterated_32_32",
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
            "DGX_DASHBOARD_ROOT": str(REPO_ROOT),
        })

        for engine, artifact in packages:
            for rank in ("0", "1"):
                completed = run([str(NODE), "preflight", engine, artifact, rank], environment)
                assert completed.returncode == 0, (engine, rank, completed.stderr)
                assert f"RANK={rank}" in completed.stdout
                assert "MODEL_REPO_HOST=" in completed.stdout
                if rank == "0":
                    assert "API_HOST=100.64.0.8" in completed.stdout
        cluster_metadata = assignments(
            packages[("sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2")] / "runtime.env",
        )
        snapshot = (
            cache
            / "hub"
            / f"models--{cluster_metadata['MODEL'].replace('/', '--')}"
            / "snapshots"
            / cluster_metadata["MODEL_REVISION"]
        )
        missing_name = "model-00001-of-00002.safetensors"
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer.0": missing_name}}),
            encoding="utf-8",
        )
        incomplete = run(
            [
                str(NODE),
                "preflight",
                "sglang",
                "unsloth_qwen36_27b_nvfp4_dflash_tp2",
                "0",
            ],
            environment,
        )
        assert incomplete.returncode == 1
        assert "has an incomplete cache snapshot" in incomplete.stderr
        assert f"references missing weight file '{missing_name}'" in incomplete.stderr
        (snapshot / missing_name).touch()


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
        same_name_environment = dict(replacement_environment)
        same_name = "inference-cluster-vllm-deepseek-v4-flash-0731-tp2-rank0"
        same_name_environment["PREFLIGHT_REPLACE_CONTAINER"] = same_name
        same_name_environment["FAKE_TARGET_CONTAINER"] = same_name
        completed = run(
            [str(NODE), "preflight", "vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "0"],
            same_name_environment,
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
            ("vllm", "drowzeys_keys_deepseek_v4_flash_dspark_abliterated_32_32", "0"),
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
            assert "CUDA_VISIBLE_DEVICES=0" in run_arguments
            assert "WORLD_SIZE=2" in run_arguments
            assert f"NODE_RANK={rank}" in run_arguments
            expected_master_port = "25000" if engine == "vllm" else "25001"
            assert f"MASTER_PORT={expected_master_port}" in run_arguments
            assert "NCCL_NET=IB" in run_arguments
            assert "NCCL_DEBUG=INFO" in run_arguments
            assert "NCCL_IB_HCA=rocep1s0f1" in run_arguments
            assert "NCCL_SOCKET_IFNAME=enp1s0f1np1" in run_arguments
            assert f"{REPO_ROOT}:{REPO_ROOT}:ro" in run_arguments
            assert run_arguments[-3:] == [str(ROOT / "runtime" / "cluster" / "serve-node.sh"), engine, artifact]
            assert any(value.endswith(":ro") and "models--" in value for value in run_arguments)
            if engine == "vllm":
                assert "VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.0" in run_arguments
                assert "KV_CACHE_DTYPE=fp8_ds_mla" in run_arguments
                assert "CLUSTER_PROFILE=quality" in run_arguments
                assert "LONG_PREFILL_TOKEN_THRESHOLD=1024" in run_arguments
                assert "DECODE_AWARE_PREFILL_INTERVAL=8" in run_arguments
                assert "DECODE_AWARE_THROTTLED_TOKENS=6" in run_arguments
                metadata = assignments(packages[(engine, artifact)] / "runtime.env")
                repository = metadata["MODEL"].replace("/", "--")
                assert any(f"MODEL_PATH=/models/hub/models--{repository}/" in value for value in run_arguments)
                assert f"SERVED={metadata['SERVED']}" in run_arguments
            else:
                assert "DRAFTER_PATH=" not in run_arguments
                assert any(value.startswith("DRAFTER_PATH=/models/hub/models--") for value in run_arguments)
                assert "SGLANG_ENABLE_SPEC_V2=1" in run_arguments
                assert "SGLANG_ENABLE_JIT_DEEPGEMM=0" in run_arguments

        profile_expectations = {
            "balanced": ("nvfp4_ds_mla", "1048576", "6", "5"),
            "throughput": ("nvfp4_ds_mla", "350000", "12", "5"),
        }
        for profile, (kv_cache, context, sequences, mtp_tokens) in profile_expectations.items():
            capture.write_text("", encoding="utf-8")
            profiled_environment = dict(environment)
            profiled_environment["CLUSTER_PROFILE"] = profile
            completed = run(
                [str(NODE), "start", "vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "0"],
                profiled_environment,
            )
            assert completed.returncode == 0, (profile, completed.stderr)
            run_arguments = next(
                arguments
                for arguments in docker_calls(capture)
                if arguments and arguments[0] == "run"
            )
            for expected in (
                f"CLUSTER_PROFILE={profile}",
                f"KV_CACHE_DTYPE={kv_cache}",
                f"MAX_MODEL_LEN={context}",
                f"MAX_NUM_SEQS={sequences}",
                f"MTP_NUM_TOKENS={mtp_tokens}",
            ):
                assert expected in run_arguments


def test_profile_is_exact_node_lifecycle_identity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        install_node_fakes(fake_bin)
        capture = temp / "docker.jsonl"
        base = base_environment(temp / "home", fake_bin)
        base.update({
            "DGX_DASHBOARD_ROOT": str(REPO_ROOT),
            "DOCKER_CAPTURE": str(capture),
            "FAKE_RUNNING_PROFILE": "throughput",
        })
        command = [
            str(NODE),
            "status",
            "vllm",
            "deepseek_ai_deepseek_v4_flash_dspark_tp2",
            "0",
        ]

        throughput = {**base, "CLUSTER_PROFILE": "throughput"}
        completed = run(command, throughput)
        assert completed.returncode == 0, completed.stderr
        assert "state=running" in completed.stdout
        assert "launch_profile=throughput" in completed.stdout
        assert "endpoint=http://100.64.0.8:8888/v1" in completed.stdout

        quality = {**base, "CLUSTER_PROFILE": "quality"}
        completed = run(command, quality)
        assert completed.returncode == 0, completed.stderr
        assert "state=absent" in completed.stdout
        assert "launch_profile=quality" in completed.stdout

        rejected = run(
            [str(NODE), "logs", "vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "0"],
            quality,
        )
        assert rejected.returncode == 1
        assert "requested lifecycle identity is not active" in rejected.stderr
        rejected = run(
            [str(NODE), "verify", "vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "0"],
            quality,
        )
        assert rejected.returncode == 1
        assert "wrong launch profile" in rejected.stderr

        capture.write_text("", encoding="utf-8")
        completed = run(
            [str(NODE), "stop", "vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "0"],
            quality,
        )
        assert completed.returncode == 0, completed.stderr
        assert not any(call and call[0] == "rm" for call in docker_calls(capture))

        capture.write_text("", encoding="utf-8")
        completed = run(
            [str(NODE), "stop", "vllm", "deepseek_ai_deepseek_v4_flash_dspark_tp2", "0"],
            throughput,
        )
        assert completed.returncode == 0, completed.stderr
        assert any(call[:2] == ["rm", "-f"] for call in docker_calls(capture))


def test_sglang_command_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        capture = temp / "command.bin"
        executable(
            fake_bin / "python3",
            "#!/bin/bash\nprintf '%s\\0' \"$@\" >\"${COMMAND_CAPTURE:?}\"\n",
        )
        environment = {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "COMMAND_CAPTURE": str(capture),
            "NODE_RANK": "1",
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "25001",
            "MODEL_PATH": "/models/model",
            "DRAFTER_PATH": "/models/drafter",
            "SERVED": "qwen36-27b-unsloth-nvfp4-dflash-tp2",
            "API_HOST": "100.64.0.8",
            "API_PORT": "8888",
            "MAX_MODEL_LEN": "262144",
            "NCCL_SOCKET_IFNAME": "enp1s0f1np1",
            "NCCL_IB_HCA": "rocep1s0f1",
        }
        completed = run(
            [str(SERVE_NODE), "sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2"],
            environment,
        )
        assert completed.returncode == 0, completed.stderr
        arguments = [value.decode() for value in capture.read_bytes().split(b"\0") if value]
        assert arguments[:2] == ["-m", "sglang.launch_server"]
        assert arguments[arguments.index("--served-model-name") + 1] == environment["SERVED"]
        assert arguments[arguments.index("--node-rank") + 1] == "1"
        assert arguments[arguments.index("--max-running-requests") + 1] == "8"
        assert arguments[arguments.index("--speculative-num-draft-tokens") + 1] == "20"
        assert arguments[arguments.index("--speculative-draft-window-size") + 1] == "4096"
        assert "--disable-cuda-graph" in arguments
        assert "--headless" not in arguments
        assert "--cuda-graph-max-bs" not in arguments


def test_sglang_worker_verification_contract() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        executable(
            fake_bin / "ss",
            "#!/bin/bash\nprintf '%s\\n' 'LISTEN 0 128 127.0.0.1:8888 0.0.0.0:*'\n",
        )
        executable(
            fake_bin / "docker",
            """#!/usr/bin/env python3
import json, os, sys
arguments = sys.argv[1:]
if arguments[:2] == ["inspect", "-f"]:
    template = arguments[2]
    if ".State.Running" in template:
        print("true")
    elif ".Config.Env" in template:
        print("\\n".join([
            "NCCL_NET=IB",
            "NCCL_IB_DISABLE=0",
            "NCCL_IB_HCA=rocep1s0f1",
            "NCCL_SOCKET_IFNAME=enp1s0f1np1",
            "GLOO_SOCKET_IFNAME=enp1s0f1np1",
            "TP_SOCKET_IFNAME=enp1s0f1np1",
            "MASTER_ADDR=10.0.0.1",
            "MASTER_PORT=25001",
            "WORLD_SIZE=2",
            "NODE_RANK=1",
        ]))
    sys.exit(0)
if arguments and arguments[0] == "inspect":
    print(json.dumps([{
        "HostConfig": {
            "Privileged": False,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges:true"],
            "ReadonlyRootfs": True,
            "PidsLimit": 4096,
            "NetworkMode": "host",
        },
        "Mounts": [],
    }]))
    sys.exit(0)
if arguments and arguments[0] == "exec":
    print(os.environ.get(
        "FAKE_COMMAND",
        "python3 -m sglang.launch_server --nnodes 2 --node-rank 1 "
        "--dist-init-addr 10.0.0.1:25001 --tp-size 2 --host 127.0.0.1",
    ))
    sys.exit(0)
if arguments and arguments[0] == "logs":
    print("topology rank=1 world_size=2 master=10.0.0.1:25001 "
          "dist_if=enp1s0f1np1 rdma_hca=rocep1s0f1")
    print("NCCL INFO NET/IB : Using [0]rocep1s0f1:1/RoCE")
    print("Channel 00/0 : 1[0] -> 0[0] [send] via NET/IB/0")
    sys.exit(0)
sys.exit(1)
""",
        )
        environment = {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(temp / "home"),
            "DGX_DASHBOARD_ROOT": str(REPO_ROOT),
        }
        completed = run(
            [str(NODE), "verify", "sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2", "1"],
            environment,
        )
        assert completed.returncode == 0, completed.stderr
        environment["FAKE_COMMAND"] = (
            "python3 -m sglang.launch_server --nnodes 2 --node-rank 1 "
            "--dist-init-addr 10.0.0.9:25001 --tp-size 2 --host 127.0.0.1"
        )
        rejected = run(
            [str(NODE), "verify", "sglang", "unsloth_qwen36_27b_nvfp4_dflash_tp2", "1"],
            environment,
        )
        assert rejected.returncode != 0
        assert "wrong distributed initialization address" in rejected.stderr


def main() -> int:
    test_cluster_front_door()
    test_profiled_cluster_front_door()
    test_transactional_failures()
    test_node_preflight_and_launch()
    test_profile_is_exact_node_lifecycle_identity()
    test_sglang_command_contract()
    test_sglang_worker_verification_contract()
    print("cluster contracts: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

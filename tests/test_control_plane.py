#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "tools" / "parse-runtime-env.py"
SERVE = ROOT / "serve.sh"
KEYS = (
    "IMAGE", "MODEL", "MODEL_REVISION", "MODEL_HOST_PATH", "SERVED",
    "DRAFTER", "DRAFTER_REVISION", "DRAFTER_HOST_PATH", "TOKENIZER",
    "TOKENIZER_REVISION", "TOKENIZER_HOST_PATH", "CONTAINER_NAME",
)
PUBLIC_IMAGE = "vllm/vllm-openai@sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
REVISION = "1" * 40


def run(command: list[str], environment: dict[str, str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def parse_assignments(path: pathlib.Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


def write_metadata(path: pathlib.Path, values: dict[str, str]) -> None:
    path.write_text("".join(f"{key}={values[key]}\n" for key in KEYS), encoding="utf-8")


def valid_values() -> dict[str, str]:
    return {
        "IMAGE": PUBLIC_IMAGE,
        "MODEL": "owner/model",
        "MODEL_REVISION": REVISION,
        "MODEL_HOST_PATH": "",
        "SERVED": "served-model",
        "DRAFTER": "",
        "DRAFTER_REVISION": "",
        "DRAFTER_HOST_PATH": "",
        "TOKENIZER": "",
        "TOKENIZER_REVISION": "",
        "TOKENIZER_HOST_PATH": "",
        "CONTAINER_NAME": "contract-container",
    }


def parser_result(path: pathlib.Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([str(PARSER), str(path)], cwd=ROOT, capture_output=True, timeout=30)


def test_metadata_parser() -> None:
    metadata_paths = sorted(ROOT.glob("serve/*/*/*/runtime.env")) + sorted(ROOT.glob("serve/cluster/*/*/runtime.env"))
    assert len(metadata_paths) == 80, len(metadata_paths)
    for path in metadata_paths:
        completed = parser_result(path)
        assert completed.returncode == 0, (path, completed.stderr.decode(errors="replace"))
        assert len(completed.stdout.split(b"\0")) == 25

    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        cases: list[tuple[str, dict[str, str], str]] = []
        values = valid_values()
        values["MODEL"] = "owner/$(id)"
        cases.append(("model-shell", values, "MODEL must be an owner/repository identifier"))
        values = valid_values()
        values["SERVED"] = "$(id)"
        cases.append(("served-shell", values, "SERVED is invalid"))
        values = valid_values()
        values["IMAGE"] = "vllm/vllm-openai:latest"
        cases.append(("latest-image", values, "IMAGE must be an immutable digest reference or a versioned local tag"))
        values = valid_values()
        values["MODEL"] = "/models/../escape"
        values["MODEL_REVISION"] = ""
        values["MODEL_HOST_PATH"] = "/host/model"
        cases.append(("dot-segment", values, "MODEL must not contain dot segments"))
        values = valid_values()
        values["MODEL"] = "owner/model with space"
        cases.append(("space", values, "MODEL must be an owner/repository identifier"))
        for name, values, message in cases:
            path = root / f"{name}.env"
            write_metadata(path, values)
            completed = parser_result(path)
            assert completed.returncode == 1, name
            assert message in completed.stderr.decode(), (name, completed.stderr)

        valid_local = valid_values()
        valid_local["IMAGE"] = "inference-workspace/runtime:v1.2.3"
        valid_local["MODEL"] = "/models/local/model"
        valid_local["MODEL_REVISION"] = ""
        valid_local["MODEL_HOST_PATH"] = "/srv/models/model"
        path = root / "valid-local.env"
        write_metadata(path, valid_local)
        assert parser_result(path).returncode == 0

        duplicate = root / "duplicate.env"
        write_metadata(duplicate, valid_values())
        duplicate.write_text(duplicate.read_text() + "MODEL=owner/other\n", encoding="utf-8")
        completed = parser_result(duplicate)
        assert completed.returncode == 1
        assert "exactly 12 assignments" in completed.stderr.decode()


def make_cache(cache: pathlib.Path, repository: str, revision: str) -> None:
    (cache / "hub" / f"models--{repository.replace('/', '--')}" / "snapshots" / revision).mkdir(parents=True)


def minimal_environment(home: pathlib.Path) -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "HOME": str(home), "LC_ALL": "C"}


def test_dispatcher() -> None:
    local_artifact = "qwen36_27b_fp8"
    local_package = ROOT / "serve" / "vllm" / "rtx6000" / local_artifact
    spark_artifact = "poolside_laguna_s_2_1_nvfp4_dflash"
    metadata = parse_assignments(local_package / "runtime.env")
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        cache = temp / "hf"
        make_cache(cache, metadata["MODEL"], metadata["MODEL_REVISION"])
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        write_fake_nvidia(fake_bin / "nvidia-smi", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition\n")
        expected_commit = "a" * 40
        (fake_bin / "git").write_text(
            "#!/usr/bin/env bash\n"
            "case \"$*\" in\n"
            "  *status*) exit 0 ;;\n"
            "  *symbolic-ref*) exit 0 ;;\n"
            "  *rev-parse*) printf '%s\\n' " + shlex.quote(expected_commit) + " ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        (fake_bin / "git").chmod(0o755)
        ssh_log = temp / "ssh.log"
        (fake_bin / "ssh").write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\0' \"$@\" >{shlex.quote(str(ssh_log))}\n"
            "printf 'remote stdout\\n'\n"
            "printf 'remote stderr\\n' >&2\n"
            "exit \"${SSH_EXIT:-0}\"\n",
            encoding="utf-8",
        )
        (fake_bin / "ssh").chmod(0o755)
        environment = minimal_environment(temp / "home")
        environment.update({
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HF_CACHE": str(cache),
            "OFFLINE": "1",
            "SERVE_DRY_RUN": "1",
            "HF_TOKEN": "do-not-forward",
        })

        completed = run(
            [str(SERVE), "local", "vllm", local_artifact, "--contract-extra", "value with spaces"],
            environment,
        )
        assert completed.returncode == 0, completed.stderr
        assert "target=local profile=rtx6000 engine=vllm" in completed.stderr
        arguments = shlex.split(completed.stdout)
        command_index = arguments.index("/run/inference/package/serve.sh")
        assert arguments[command_index + 1:] == ["--contract-extra", "value with spaces"]

        help_result = run([str(SERVE), "--help"], environment)
        assert help_result.returncode == 0
        for catalog in (
            "available vllm/rtx6000 artifacts:",
            "available sglang/rtx6000 artifacts:",
            "available vllm/spark artifacts:",
            "available sglang/spark artifacts:",
            "available cluster/vllm artifacts:",
            "available cluster/sglang artifacts:",
        ):
            assert catalog in help_result.stdout
        assert "docker " not in help_result.stdout
        assert not ssh_log.exists()

        for target, host in (("spark1", "spark1-ts"), ("spark2", "spark2-ts"), ("spark3", "spark3-ts")):
            remote = run(
                [str(SERVE), target, "vllm", spark_artifact, "--label", "value with spaces"],
                {**environment, "HF_CACHE": "", "ATTENTION_BACKEND": "FLASH INFER"},
            )
            assert remote.returncode == 0, remote.stderr
            assert remote.stdout == "remote stdout\n"
            assert "remote stderr\n" in remote.stderr
            assert f"target={target} host={host} profile=spark engine=vllm" in remote.stderr
            ssh_arguments = ssh_log.read_bytes().split(b"\0")[:-1]
            assert ssh_arguments[6].decode() == host
            remote_command = ssh_arguments[7].decode()
            assert expected_commit in remote_command
            assert "HF_CACHE=" in remote_command
            assert "ATTENTION_BACKEND=FLASH\\ INFER" in remote_command
            assert "value\\ with\\ spaces" in remote_command
            assert "HF_TOKEN" not in remote_command

        failed_ssh = run(
            [str(SERVE), "spark1", "vllm", spark_artifact],
            {**environment, "SSH_EXIT": "23"},
        )
        assert failed_ssh.returncode == 23
        assert failed_ssh.stdout == "remote stdout\n"
        assert "remote stderr\n" in failed_ssh.stderr

        for command in (
            [str(SERVE), local_artifact],
            [str(SERVE), "vllm", local_artifact],
            [str(SERVE), "local", local_artifact],
            [str(SERVE), "local", "llama.cpp", local_artifact],
            [str(SERVE), "unknown", "vllm", local_artifact],
        ):
            ssh_log.unlink(missing_ok=True)
            rejected = run(command, environment)
            assert rejected.returncode == 2, (command, rejected.stderr)
            assert "usage:" in rejected.stderr
            assert not ssh_log.exists()

        unknown = run([str(SERVE), "local", "vllm", "does_not_exist"], environment)
        assert unknown.returncode == 1
        assert "error: no rtx6000 vllm recipe named 'does_not_exist'" in unknown.stderr

        for artifact in (".", "..", "../escape", "/tmp/escape", "bad.name", "bad/name"):
            rejected = run([str(SERVE), "local", "vllm", artifact], environment)
            assert rejected.returncode == 1
            assert "invalid artifact name" in rejected.stderr

        outside = temp / "outside-package"
        outside.mkdir()
        write_metadata(outside / "runtime.env", valid_values())
        (outside / "serve.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        escape = ROOT / "serve" / "vllm" / "rtx6000" / "escapecontract"
        escape.symlink_to(outside, target_is_directory=True)
        try:
            rejected = run([str(SERVE), "local", "vllm", "escapecontract"], environment)
            assert rejected.returncode == 1
            assert "escapes" in rejected.stderr
        finally:
            escape.unlink()


def write_fake_nvidia(path: pathlib.Path, output: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nprintf '%s' {shlex.quote(output)}\n", encoding="utf-8")
    path.chmod(0o755)


def test_gpu_detection() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        environment = minimal_environment(temp / "home")
        environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"
        command = [str(SERVE), "local", "vllm", "qwen36_27b_fp8"]
        for output in ("NVIDIA GB10\n", "NVIDIA GB10\nNVIDIA GB10\n", "Unknown GPU\n", ""):
            write_fake_nvidia(fake_bin / "nvidia-smi", output)
            rejected = run(command, environment)
            assert rejected.returncode == 1
            assert "single-device commands must be initiated from the RTX workstation" in rejected.stderr

        help_result = run([str(SERVE), "--help"], environment)
        assert help_result.returncode == 0
        assert "single-device commands must be initiated" not in help_result.stderr


def make_test_package(root: pathlib.Path, values: dict[str, str], *, builder: bool = False) -> pathlib.Path:
    package = root / "package"
    package.mkdir()
    write_metadata(package / "runtime.env", values)
    serve = package / "serve.sh"
    serve.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    serve.chmod(0o755)
    if builder:
        build = package / "build-image.sh"
        build.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        build.chmod(0o755)
    return package


def resolver_run(package: pathlib.Path, cache: pathlib.Path, home: pathlib.Path, extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    environment = minimal_environment(home)
    environment.update({"HF_CACHE": str(cache), "OFFLINE": "1", "SERVE_DRY_RUN": "1"})
    if extra:
        environment.update(extra)
    return run([str(ROOT / "runtime" / "rtx6000" / "run_vllm_docker.sh"), str(package)], environment)


def test_resolver_failures() -> None:
    labels = ("model", "drafter", "tokenizer")
    for missing_label in labels:
        with tempfile.TemporaryDirectory() as temporary:
            temp = pathlib.Path(temporary)
            values = valid_values()
            values["DRAFTER"] = "owner/drafter"
            values["DRAFTER_REVISION"] = "2" * 40
            values["TOKENIZER"] = "owner/tokenizer"
            values["TOKENIZER_REVISION"] = "3" * 40
            package = make_test_package(temp, values)
            cache = temp / "cache"
            ordered = (("MODEL", "model"), ("DRAFTER", "drafter"), ("TOKENIZER", "tokenizer"))
            for label, lower in ordered:
                if lower == missing_label:
                    break
                make_cache(cache, values[label], values[f"{label}_REVISION"])
            completed = resolver_run(package, cache, temp / "home")
            assert completed.returncode == 1
            upper = missing_label.upper()
            expected = f"{missing_label} '{values[upper]}@{values[f'{upper}_REVISION']}' is not installed in any configured cache root"
            assert expected in completed.stderr, (missing_label, completed.stderr)

    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        values = valid_values()
        package = make_test_package(temp, values)
        default_cache = temp / "home" / ".cache" / "huggingface"
        make_cache(default_cache, values["MODEL"], values["MODEL_REVISION"])
        completed = resolver_run(package, temp / "explicit-empty", temp / "home")
        assert completed.returncode == 1
        assert "model 'owner/model@" in completed.stderr

    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        values = valid_values()
        values["IMAGE"] = "inference-workspace/test-runtime:v1"
        package = make_test_package(temp, values, builder=True)
        cache = temp / "cache"
        make_cache(cache, values["MODEL"], values["MODEL_REVISION"])
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        docker = fake_bin / "docker"
        docker.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
        docker.chmod(0o755)
        environment = minimal_environment(temp / "home")
        environment.update({"PATH": f"{fake_bin}:/usr/bin:/bin", "HF_CACHE": str(cache), "OFFLINE": "0"})
        completed = run([str(ROOT / "runtime" / "rtx6000" / "run_vllm_docker.sh"), str(package)], environment)
        assert completed.returncode == 1
        expected = f"image 'inference-workspace/test-runtime:v1' is unavailable; run '{package}/build-image.sh'"
        assert expected in completed.stderr


def test_single_preflight() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        values = valid_values()
        package = make_test_package(temp, values)
        cache = temp / "cache"
        make_cache(cache, values["MODEL"], values["MODEL_REVISION"])
        fake_bin = temp / "bin"
        fake_bin.mkdir()
        marker = temp / "container-created"
        old_image = "sha256:" + "2" * 64
        docker = fake_bin / "docker"
        docker.write_text(
            """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
if arguments[:2] == ["image", "inspect"]:
    raise SystemExit(0)
if arguments[:2] == ["container", "inspect"]:
    name = arguments[2]
    if name == "contract-container":
        raise SystemExit(1)
    if name == "old-production":
        print(json.dumps([{
            "State": {"Running": True},
            "Image": os.environ["FAKE_OLD_IMAGE_ID"],
            "HostConfig": {
                "PortBindings": {
                    "8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8000"}]
                }
            },
        }]))
        raise SystemExit(0)
    raise SystemExit(1)
if arguments and arguments[0] == "run":
    Path(os.environ["CONTAINER_MARKER"]).write_text("created", encoding="utf-8")
raise SystemExit(90)
""",
            encoding="utf-8",
        )
        docker.chmod(0o755)
        environment = minimal_environment(temp / "home")
        environment.update({
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HF_CACHE": str(cache),
            "OFFLINE": "1",
            "PREFLIGHT_ONLY": "1",
            "PREFLIGHT_REPLACE_CONTAINER": "old-production",
            "PREFLIGHT_REPLACE_IMAGE_ID": old_image,
            "FAKE_OLD_IMAGE_ID": old_image,
            "CONTAINER_MARKER": str(marker),
        })
        completed = run(
            [str(ROOT / "runtime" / "rtx6000" / "run_vllm_docker.sh"), str(package)],
            environment,
        )
        assert completed.returncode == 0, completed.stderr
        assert "docker run " in completed.stdout
        assert "preflight complete; no container created" in completed.stdout
        assert not marker.exists()

        mismatched = dict(environment)
        mismatched["PREFLIGHT_REPLACE_IMAGE_ID"] = "sha256:" + "3" * 64
        rejected = run(
            [str(ROOT / "runtime" / "rtx6000" / "run_vllm_docker.sh"), str(package)],
            mismatched,
        )
        assert rejected.returncode == 1
        assert "replacement container 'old-production' image is" in rejected.stderr
        assert not marker.exists()


def test_cluster_dispatcher_actions() -> None:
    package = ROOT / "serve" / "cluster" / "vllm" / "contractcluster"
    package.mkdir()
    try:
        write_metadata(package / "runtime.env", valid_values())
        for action in ("start", "status", "logs", "verify", "stop"):
            script = package / f"{action}.sh"
            script.write_text(
                f"#!/usr/bin/env bash\nprintf '%s\\n' {shlex.quote(action)}\nprintf '<%s>\\n' \"$@\"\n",
                encoding="utf-8",
            )
            script.chmod(0o755)
        environment = minimal_environment(pathlib.Path("/tmp/unused-home"))
        for arguments, expected in (
            ([], "start"),
            (["start"], "start"),
            (["status"], "status"),
            (["logs"], "logs"),
            (["verify"], "verify"),
            (["stop"], "stop"),
        ):
            completed = run([str(SERVE), "cluster", "vllm", "contractcluster", *arguments], environment)
            assert completed.returncode == 0, completed.stderr
            assert completed.stdout.splitlines()[0] == expected
            assert f"action={expected}" in completed.stderr
    finally:
        for child in package.iterdir():
            child.unlink()
        package.rmdir()


def test_remote_single_validation() -> None:
    source = (ROOT / "runtime" / "spark" / "run-remote-single.sh").read_text(encoding="utf-8")
    expected_commit = "a" * 40
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        checkout = temp / "inference"
        checkout.mkdir()
        (checkout / ".git").mkdir()
        package = checkout / "serve" / "vllm" / "spark" / "artifact"
        package.mkdir(parents=True)
        write_metadata(package / "runtime.env", valid_values())
        (package / "serve.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        runtime = checkout / "runtime" / "spark"
        runtime.mkdir(parents=True)
        marker = temp / "docker-marker"
        runner = runtime / "run_vllm_docker.sh"
        runner.write_text(
            f"#!/usr/bin/env bash\nprintf '%s\\0' \"$@\" >{shlex.quote(str(marker))}\n",
            encoding="utf-8",
        )
        runner.chmod(0o755)
        remote = temp / "run-remote-single.sh"
        remote.write_text(source.replace("ROOT=/home/jjlink/inference", f"ROOT={shlex.quote(str(checkout))}"), encoding="utf-8")
        remote.chmod(0o755)

        fake_bin = temp / "bin"
        fake_bin.mkdir()
        git = fake_bin / "git"
        git.write_text(
            "#!/usr/bin/env bash\n"
            "case \"$*\" in\n"
            "  *symbolic-ref*) [[ ${GIT_STATE:-clean} != detached ]] ;;\n"
            "  *status*) [[ ${GIT_STATE:-clean} == dirty ]] && printf ' M changed\\n' ;;\n"
            f"  *rev-parse*) [[ ${{GIT_STATE:-clean}} == mismatch ]] && printf '%s\\n' {'b' * 40} || printf '%s\\n' {expected_commit} ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        git.chmod(0o755)
        write_fake_nvidia(fake_bin / "nvidia-smi", "NVIDIA GB10\n")
        environment = minimal_environment(temp / "home")
        environment["PATH"] = f"{fake_bin}:/usr/bin:/bin"
        command = [str(remote), expected_commit, "vllm", "artifact", "--label", "value with spaces"]

        completed = run(command, environment)
        assert completed.returncode == 0, completed.stderr
        assert marker.read_bytes().split(b"\0")[-2] == b"value with spaces"

        for updates, message in (
            ({"GIT_STATE": "dirty"}, "is dirty"),
            ({"GIT_STATE": "detached"}, "is detached"),
            ({"GIT_STATE": "mismatch"}, "expected"),
        ):
            marker.unlink(missing_ok=True)
            rejected = run(command, {**environment, **updates})
            assert rejected.returncode == 1
            assert message in rejected.stderr
            assert not marker.exists()

        marker.unlink(missing_ok=True)
        write_fake_nvidia(fake_bin / "nvidia-smi", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition\n")
        rejected = run(command, environment)
        assert rejected.returncode == 1
        assert "exactly one NVIDIA GB10" in rejected.stderr
        assert not marker.exists()

        write_fake_nvidia(fake_bin / "nvidia-smi", "NVIDIA GB10\n")
        rejected = run([str(remote), expected_commit, "vllm", "missing"], environment)
        assert rejected.returncode == 1
        assert "no spark vllm recipe" in rejected.stderr
        assert not marker.exists()

        outside = temp / "outside"
        outside.mkdir()
        write_metadata(outside / "runtime.env", valid_values())
        (outside / "serve.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        (package.parent / "escape").symlink_to(outside, target_is_directory=True)
        rejected = run([str(remote), expected_commit, "vllm", "escape"], environment)
        assert rejected.returncode == 1
        assert "escapes" in rejected.stderr
        assert not marker.exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("all", "dispatcher", "runtime"), nargs="?", default="all")
    arguments = parser.parse_args()
    if arguments.group in {"all", "runtime"}:
        test_metadata_parser()
        test_resolver_failures()
        test_single_preflight()
        test_remote_single_validation()
    if arguments.group in {"all", "dispatcher"}:
        test_dispatcher()
        test_gpu_detection()
        test_cluster_dispatcher_actions()
    print(f"control-plane {arguments.group}: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

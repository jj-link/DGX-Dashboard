#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "tools" / "parse-runtime-env.py"
FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "recipe-contracts"
CATALOGS = {('rtx6000', 'vllm'): ['aeon_qwen36_27b_nvfp4', 'bottlecapai_thinkingcap_qwen36_27b_fp8', 'deepseek_v4_flash_tp2', 'gemma4_12b_awq', 'gemma4_31b_nvfp4', 'lfm25_8b_a1b_prismaquant_65bit', 'morosystems_thinkingcap_qwen36_27b_nvfp4', 'nvidia_qwen36_27b_nvfp4', 'nvidia_qwen36_27b_nvfp4_dflash11', 'nvidia_qwen36_27b_nvfp4_dflash11_bf16kv', 'nvidia_qwen36_27b_nvfp4_mtp3_bf16kv', 'prismascout_qwen36_27b_nvfp4', 'qwen35_122b_a10b_nvfp4', 'qwen36_27b_fp8', 'qwen36_27b_fp8_dflash', 'qwen36_a3b_fp8', 'qwen36_a3b_fp8_dflash', 'qwen36_a3b_fp8_mtp', 'qwen36_a3b_nvfp4', 'qwen36_nvfp4_base', 'qwen36_nvfp4_mtp', 'unsloth_qwen36_27b_nvfp4', 'unsloth_qwen36_35b_a3b_nvfp4_fast'], ('rtx6000', 'sglang'): ['gemma4_12b_awq', 'gemma4_31b_dflash', 'nvidia_qwen36_27b_nvfp4', 'nvidia_qwen36_27b_nvfp4_dflash', 'qwen25_7b_awq', 'qwen36_27b_fp8_dflash', 'qwen36_27b_fp8_eagle3', 'qwen36_a3b_fp8_dflash', 'unsloth_qwen36_27b_nvfp4', 'unsloth_qwen36_35b_a3b_nvfp4_fast'], ('spark', 'vllm'): ['aeon7_qwen36_27b_aeon_ultimate_uncensored_nvfp4', 'axionml_gemma4_12b_nvfp4', 'deepseek_ai_deepseek_v4_flash_dspark', 'melcheikh_gemma4_31b_it_qat_nvfp4_blackwell', 'morosystems_thinkingcap_qwen36_27b_nvfp4_dflash', 'nvidia_diffusiongemma_26b_a4b_it_nvfp4', 'nvidia_gemma4_26b_a4b_nvfp4', 'nvidia_qwen36_35b_a3b_nvfp4', 'poolside_laguna_s_2_1_nvfp4_dflash', 'qwen36_27b_bf16', 'qwen36_27b_bf16_dflash', 'qwen36_27b_fp8', 'qwen36_27b_fp8_dflash', 'qwen36_35b_a3b_bf16', 'qwen36_35b_a3b_bf16_dflash', 'qwen36_35b_a3b_fp8', 'qwen36_35b_a3b_fp8_dflash', 'redhatai_qwen35_122b_a10b_nvfp4', 'redhatai_qwen36_35b_a3b_nvfp4', 'redhatai_qwen36_35b_a3b_nvfp4_dflash', 'unsloth_qwen36_27b_nvfp4', 'unsloth_qwen36_27b_nvfp4_dflash', 'unsloth_qwen36_35b_a3b_nvfp4_fast_mtp3'], ('spark', 'sglang'): ['axionml_gemma4_12b_nvfp4_dflash', 'luni_ornith_1_0_9b_nvfp4_awq', 'nvidia_diffusiongemma_26b_a4b_it_nvfp4', 'nvidia_gemma4_26b_a4b_nvfp4', 'nvidia_gemma4_31b_it_nvfp4_dflash', 'nvidia_qwen36_27b_nvfp4', 'nvidia_qwen36_27b_nvfp4_dflash', 'nvidia_qwen36_35b_a3b_nvfp4', 'qwen36_27b_fp8', 'qwen36_27b_fp8_dflash', 'qwen36_27b_fp8_eagle3', 'qwen36_35b_a3b_fp8', 'qwen36_35b_a3b_fp8_dflash', 'r0b0tlab_nex_n2_mini_nvfp4', 'redhatai_diffusiongemma_26b_a4b_it_nvfp4', 'redhatai_qwen36_35b_a3b_nvfp4', 'redhatai_qwen36_35b_a3b_nvfp4_dflash', 'sakamakismile_ornith_1_0_35b_nvfp4', 'unsloth_qwen36_27b_nvfp4', 'unsloth_qwen36_27b_nvfp4_dflash']}
KEYS = (
    "IMAGE", "MODEL", "MODEL_REVISION", "MODEL_HOST_PATH", "SERVED",
    "DRAFTER", "DRAFTER_REVISION", "DRAFTER_HOST_PATH", "TOKENIZER",
    "TOKENIZER_REVISION", "TOKENIZER_HOST_PATH", "CONTAINER_NAME",
)


def run(command: list[str], *, env: dict[str, str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def load_metadata(path: pathlib.Path) -> dict[str, str]:
    completed = subprocess.run(
        [str(PARSER), str(path)],
        cwd=ROOT,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode:
        raise AssertionError(f"metadata rejected for {path}: {completed.stderr.decode(errors='replace')}")
    fields = completed.stdout.split(b"\0")
    if fields[-1] == b"":
        fields.pop()
    assert len(fields) == 24, (path, len(fields))
    values = {
        fields[index].decode(): fields[index + 1].decode()
        for index in range(0, len(fields), 2)
    }
    assert tuple(values) == KEYS
    return values


def packages() -> list[tuple[str, str, str, pathlib.Path]]:
    found: list[tuple[str, str, str, pathlib.Path]] = []
    for (profile, engine), expected in CATALOGS.items():
        root = ROOT / "serve" / engine / profile
        actual = sorted(path.parent.name for path in root.glob("*/runtime.env"))
        assert actual == expected, f"{profile}/{engine} catalog mismatch\nexpected={expected}\nactual={actual}"
        found.extend((profile, engine, artifact, root / artifact) for artifact in actual)
    return sorted(found)


def repository_container_path(value: str, revision: str) -> str:
    return f"/models/hub/models--{value.replace('/', '--')}/snapshots/{revision}"


def artifact_container_path(metadata: dict[str, str], label: str) -> str:
    value = metadata[label]
    if not value:
        return ""
    if metadata[f"{label}_HOST_PATH"]:
        return value
    return repository_container_path(value, metadata[f"{label}_REVISION"])


def normalize(value: str, replacements: dict[str, str]) -> str:
    for source, destination in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        value = value.replace(source, destination)
    return value


def make_fake_engine_bin(temp: pathlib.Path) -> pathlib.Path:
    fake_bin = temp / "engine-bin"
    fake_bin.mkdir()
    helper = temp / "capture-engine.py"
    helper.write_text(
        """import json
import os
import pathlib
import sys
prefixes = ("CUDA_", "FLASHINFER_", "HF_", "NCCL_", "PYTORCH_", "SGLANG_", "VLLM_")
exact = {"MODEL_PATH", "DRAFTER_PATH", "TOKENIZER_PATH", "SERVED"}
environment = {key: value for key, value in os.environ.items() if key in exact or key.startswith(prefixes)}
pathlib.Path(os.environ["CAPTURE_PATH"]).write_text(json.dumps({
    "executable": pathlib.Path(sys.argv[1]).name,
    "argv": sys.argv[2:],
    "environment": environment,
}, sort_keys=True))
""",
        encoding="utf-8",
    )
    capture_call = f'exec /usr/bin/python3 "{helper}" "$0" "$@"\n'
    python_wrapper = (
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == -m && ( \"${2:-}\" == sglang.launch_server || "
        "\"${2:-}\" == vllm.entrypoints.openai.api_server ) ]]; then\n"
        f"  {capture_call}"
        "fi\n"
        "exec /usr/bin/python3 \"$@\"\n"
    )
    wrappers = {
        "python3": python_wrapper,
        "vllm": "#!/usr/bin/env bash\n" + capture_call,
    }
    for name, content in wrappers.items():
        path = fake_bin / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
    return fake_bin


def capture_engine(
    package: pathlib.Path,
    metadata: dict[str, str],
    temp: pathlib.Path,
    extra_args: tuple[str, ...] = (),
) -> dict[str, object]:
    temp.mkdir(parents=True, exist_ok=True)
    fake_bin = make_fake_engine_bin(temp)
    replacements: dict[str, str] = {}
    artifact_paths: dict[str, str] = {}
    for label in ("MODEL", "DRAFTER", "TOKENIZER"):
        if not metadata[label]:
            artifact_paths[label] = ""
            continue
        prepared = temp / f"artifact-{label.lower()}"
        prepared.mkdir()
        (prepared / "config.json").write_text("{}\n", encoding="utf-8")
        artifact_paths[label] = str(prepared)
        replacements[str(prepared)] = artifact_container_path(metadata, label)
    capture_path = temp / "engine.json"
    environment = {
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "HOME": str(temp / "home"),
        "LC_ALL": "C",
        "CAPTURE_PATH": str(capture_path),
        "MODEL_PATH": artifact_paths["MODEL"],
        "DRAFTER_PATH": artifact_paths["DRAFTER"],
        "TOKENIZER_PATH": artifact_paths["TOKENIZER"],
        "SERVED": metadata["SERVED"],
    }
    completed = run(["bash", str(package / "serve.sh"), *extra_args], env=environment)
    assert completed.returncode == 0, (
        package.relative_to(ROOT), completed.returncode, completed.stdout, completed.stderr
    )
    assert capture_path.is_file(), f"engine command not captured for {package.relative_to(ROOT)}"
    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    captured["argv"] = [normalize(value, replacements) for value in captured["argv"]]
    captured["environment"] = {
        key: normalize(value, replacements)
        for key, value in captured["environment"].items()
    }
    return captured


def create_shadow_package(
    package: pathlib.Path,
    metadata: dict[str, str],
    temp: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, dict[str, str]]:
    shadow = temp / "shadow-package"
    shadow.mkdir()
    shutil.copy2(package / "serve.sh", shadow / "serve.sh")
    fake_cache = temp / "hf"
    replacements = {str(shadow): "<PACKAGE>", str(fake_cache): "<HF_CACHE>"}
    shadow_values = dict(metadata)
    for label in ("MODEL", "DRAFTER", "TOKENIZER"):
        if not metadata[label]:
            continue
        host_key = f"{label}_HOST_PATH"
        if metadata[host_key]:
            host_path = temp / f"host-{label.lower()}"
            host_path.mkdir()
            shadow_values[host_key] = str(host_path)
            replacements[str(host_path)] = f"<{host_key}>"
        else:
            repository = f"models--{metadata[label].replace('/', '--')}"
            (fake_cache / "hub" / repository / "snapshots" / metadata[f"{label}_REVISION"]).mkdir(parents=True)
    (shadow / "runtime.env").write_text(
        "".join(f"{key}={shadow_values[key]}\n" for key in KEYS),
        encoding="utf-8",
    )
    return shadow, fake_cache, replacements


def capture_docker(
    profile: str,
    engine: str,
    package: pathlib.Path,
    metadata: dict[str, str],
    temp: pathlib.Path,
    extra_args: tuple[str, ...] = (),
    control: dict[str, str] | None = None,
) -> list[str]:
    shadow, fake_cache, replacements = create_shadow_package(package, metadata, temp)
    runtime_bin = temp / "runtime-bin"
    runtime_bin.mkdir()
    tailscale = runtime_bin / "tailscale"
    tailscale.write_text("#!/usr/bin/env bash\nprintf '100.64.0.1\\n'\n", encoding="utf-8")
    tailscale.chmod(0o755)
    environment = {
        "PATH": f"{runtime_bin}:/usr/bin:/bin",
        "HOME": str(temp / "home"),
        "LC_ALL": "C",
        "HF_CACHE": str(fake_cache),
        "OFFLINE": "1",
        "SERVE_DRY_RUN": "1",
    }
    if control:
        environment.update(control)
    wrapper = ROOT / "runtime" / profile / f"run_{engine}_docker.sh"
    completed = run([str(wrapper), str(shadow), *extra_args], env=environment)
    assert completed.returncode == 0, (
        package.relative_to(ROOT), completed.returncode, completed.stdout, completed.stderr
    )
    lines = [line for line in completed.stdout.splitlines() if line]
    assert len(lines) == 1 and lines[0].startswith("docker "), (package, lines)
    return [normalize(value, replacements) for value in shlex.split(lines[0])]


def option_value(arguments: list[str], option: str) -> str:
    index = arguments.index(option)
    return arguments[index + 1]


def assert_engine_contract(
    package: pathlib.Path,
    metadata: dict[str, str],
    captured: dict[str, object],
) -> None:
    arguments = captured["argv"]
    assert isinstance(arguments, list)
    assert option_value(arguments, "--host") == "0.0.0.0", package
    assert option_value(arguments, "--port") == "8000", package
    model_path = artifact_container_path(metadata, "MODEL")
    assert any(model_path in value for value in arguments), (package, model_path)
    assert any(metadata["SERVED"] in value for value in arguments), (package, metadata["SERVED"])


def assert_docker_contract(
    profile: str,
    package: pathlib.Path,
    metadata: dict[str, str],
    arguments: list[str],
) -> None:
    assert arguments[:2] == ["docker", "run"], package
    assert "--rm" in arguments and "--detach" not in arguments, package
    assert option_value(arguments, "--name") == metadata["CONTAINER_NAME"], package
    assert option_value(arguments, "--cap-drop") == "ALL", package
    assert option_value(arguments, "--security-opt") == "no-new-privileges:true", package
    assert "--read-only" in arguments and "--privileged" not in arguments, package
    assert option_value(arguments, "--pids-limit") == "4096", package
    expected_publish = ("127.0.0.1" if profile == "rtx6000" else "100.64.0.1") + ":8000:8000/tcp"
    assert option_value(arguments, "--publish") == expected_publish, package
    mounts = [arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == "--mount"]
    assert "type=bind,src=<PACKAGE>,dst=/run/inference/package,readonly" in mounts, package
    assert any(value.startswith("type=volume,") and value.endswith(",dst=/root/.cache") for value in mounts), package
    for label in ("MODEL", "DRAFTER", "TOKENIZER"):
        if not metadata[label]:
            continue
        if metadata[f"{label}_HOST_PATH"]:
            target = metadata[label]
        else:
            target = f"/models/hub/models--{metadata[label].replace('/', '--')}"
        assert any(f",dst={target}," in mount for mount in mounts), (package, label, target)
    for mount in mounts:
        if mount.startswith("type=bind,"):
            assert mount.endswith(",readonly"), (package, mount)
    assert "HF_HUB_OFFLINE=1" in arguments and "TRANSFORMERS_OFFLINE=1" in arguments, package
    assert metadata["IMAGE"] in arguments, package
    command_index = arguments.index("/run/inference/package/serve.sh")
    assert arguments[command_index - 1] == metadata["IMAGE"], package


def fixture_path(profile: str, engine: str, artifact: str) -> pathlib.Path:
    return FIXTURE_ROOT / profile / engine / f"{artifact}.json"


def build_fixture(profile: str, engine: str, artifact: str, package: pathlib.Path) -> dict[str, object]:
    metadata = load_metadata(package / "runtime.env")
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        engine_capture = capture_engine(package, metadata, temp / "engine")
    with tempfile.TemporaryDirectory() as temporary:
        temp = pathlib.Path(temporary)
        temp.mkdir(exist_ok=True)
        docker_arguments = capture_docker(profile, engine, package, metadata, temp)
    assert_engine_contract(package, metadata, engine_capture)
    assert_docker_contract(profile, package, metadata, docker_arguments)
    return {
        "artifact": artifact,
        "engine": engine,
        "profile": profile,
        "metadata": metadata,
        "engine_command": engine_capture,
        "docker_argv": docker_arguments,
    }


def check_extra_arguments(entries: list[tuple[str, str, str, pathlib.Path]]) -> None:
    sentinels = ("--contract-extra", "value with spaces")
    for profile, engine, artifact, package in entries:
        metadata = load_metadata(package / "runtime.env")
        with tempfile.TemporaryDirectory() as temporary:
            temp = pathlib.Path(temporary)
            default = capture_engine(package, metadata, temp / "default")
        with tempfile.TemporaryDirectory() as temporary:
            temp = pathlib.Path(temporary)
            extra = capture_engine(package, metadata, temp / "extra", sentinels)
        assert extra["argv"] == [*default["argv"], *sentinels], (profile, engine, artifact)


def check_runtime_controls(entries: list[tuple[str, str, str, pathlib.Path]]) -> None:
    profile, engine, _, package = next(item for item in entries if item[:3] == ("rtx6000", "vllm", "qwen36_27b_fp8"))
    metadata = load_metadata(package / "runtime.env")
    sentinels = ("--runtime-extra", "value with spaces")
    with tempfile.TemporaryDirectory() as temporary:
        arguments = capture_docker(
            profile,
            engine,
            package,
            metadata,
            pathlib.Path(temporary),
            sentinels,
            {
                "DETACH": "1",
                "KEEP": "1",
                "RESTART_POLICY": "unless-stopped",
                "MAXLEN": "12345",
            },
        )
    assert "--rm" not in arguments
    assert "--detach" in arguments
    assert option_value(arguments, "--restart") == "unless-stopped"
    assert "MAXLEN=12345" in arguments
    index = arguments.index("/run/inference/package/serve.sh")
    assert arguments[index + 1:] == list(sentinels)


def check_fixture_set(update: bool) -> None:
    entries = packages()
    expected_paths: set[pathlib.Path] = set()
    for profile, engine, artifact, package in entries:
        fixture = build_fixture(profile, engine, artifact, package)
        path = fixture_path(profile, engine, artifact)
        expected_paths.add(path)
        rendered = json.dumps(fixture, indent=2, sort_keys=True) + "\n"
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
        else:
            assert path.is_file(), f"missing fixture {path.relative_to(ROOT)}"
            assert path.read_text(encoding="utf-8") == rendered, f"fixture drift: {path.relative_to(ROOT)}"
    actual_paths = set(FIXTURE_ROOT.glob("*/*/*.json")) if FIXTURE_ROOT.exists() else set()
    if update:
        for stale in actual_paths - expected_paths:
            stale.unlink()
    else:
        assert actual_paths == expected_paths, "recipe fixture set does not match the package catalog"
    check_extra_arguments(entries)
    check_runtime_controls(entries)
    print(f"recipe contracts: {len(entries)} packages")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-fixtures", action="store_true")
    arguments = parser.parse_args()
    check_fixture_set(arguments.update_fixtures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

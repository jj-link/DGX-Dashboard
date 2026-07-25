"""CPU-safe dependency and launch-argument smoke checks for the patched image."""

import importlib.metadata
import os
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


EXPECTED_FLASHINFER_VERSION = "0.6.14"


def test_flashinfer_components_are_aligned():
    versions = {
        distribution: Version(importlib.metadata.version(distribution))
        for distribution in (
            "flashinfer-python",
            "flashinfer-cubin",
            "flashinfer-jit-cache",
        )
    }

    assert {
        distribution: version.base_version
        for distribution, version in versions.items()
    } == {
        "flashinfer-python": EXPECTED_FLASHINFER_VERSION,
        "flashinfer-cubin": EXPECTED_FLASHINFER_VERSION,
        "flashinfer-jit-cache": EXPECTED_FLASHINFER_VERSION,
    }
    assert versions["flashinfer-jit-cache"].local == "cu130"


def test_pinned_source_dependency_metadata_matches_runtime():
    requirements = {
        canonicalize_name(requirement.name): requirement
        for requirement in (
            Requirement(value)
            for value in importlib.metadata.distribution("sglang").requires or ()
        )
    }

    for distribution in (
        "flashinfer-python",
        "sgl-deep-gemm",
        "sglang-kernel",
        "torch",
    ):
        requirement = requirements[canonicalize_name(distribution)]
        installed_version = Version(importlib.metadata.version(distribution))
        assert installed_version in requirement.specifier, (
            distribution,
            installed_version,
            requirement,
        )


def test_launch_server_help_is_cpu_safe_and_complete():
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-m", "sglang.launch_server", "--help"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--model-path" in completed.stdout
    assert "--quantization" in completed.stdout
    assert "--port" in completed.stdout

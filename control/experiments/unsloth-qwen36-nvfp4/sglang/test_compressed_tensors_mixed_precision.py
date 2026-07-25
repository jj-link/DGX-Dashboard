"""Regression tests for mixed FP8/NVFP4 Unsloth Qwen3.6 configs.

The two config fixtures preserve the quantization config-group shapes from the
pinned local checkpoints. They intentionally omit unrelated model metadata and
the long ignore lists.
"""

from copy import deepcopy

import pytest
import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4_moe,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
    CompressedTensorsW4A4Nvfp4MoE,
    CompressedTensorsW8A8Fp8,
)


DENSE_REVISION = "9c295353cf85c4047fb45c27847e6c4b58596f3f"
MOE_REVISION = "11fb1afff493ad0d94c808ceecdf73b8215ccc7a"

FP8_WEIGHTS = {
    "actorder": None,
    "block_structure": None,
    "dynamic": False,
    "group_size": None,
    "num_bits": 8,
    "observer": "memoryless_minmax",
    "observer_kwargs": {},
    "scale_dtype": None,
    "strategy": "channel",
    "symmetric": True,
    "type": "float",
    "zp_dtype": None,
}
FP8_INPUT_ACTIVATIONS = {
    "actorder": None,
    "block_structure": None,
    "dynamic": True,
    "group_size": None,
    "num_bits": 8,
    "observer": None,
    "observer_kwargs": {},
    "scale_dtype": None,
    "strategy": "token",
    "symmetric": True,
    "type": "float",
    "zp_dtype": None,
}
NVFP4_INPUT_ACTIVATIONS = {
    "actorder": None,
    "block_structure": None,
    "dynamic": "local",
    "group_size": 16,
    "num_bits": 4,
    "observer": "static_minmax",
    "observer_kwargs": {},
    "scale_dtype": "torch.float8_e4m3fn",
    "strategy": "tensor_group",
    "symmetric": True,
    "type": "float",
    "zp_dtype": None,
}

DENSE_CONFIG = {
    "format": "mixed-precision",
    "global_compression_ratio": None,
    "ignore": [],
    "config_groups": {
        "group_0": {
            "format": "float-quantized",
            "input_activations": FP8_INPUT_ACTIVATIONS,
            "output_activations": None,
            "targets": [
                r"re:.*self_attn\.(q|k|v|o)_proj$",
                r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                r"re:.*lm_head",
                r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
            ],
            "weights": FP8_WEIGHTS,
        },
        "group_1": {
            "format": "nvfp4-pack-quantized",
            "input_activations": NVFP4_INPUT_ACTIVATIONS,
            "output_activations": None,
            "targets": [r"re:.*mlp\.(gate|up|down)_proj$"],
            "weights": {
                "actorder": "static",
                "block_structure": None,
                "dynamic": False,
                "group_size": 16,
                "num_bits": 4,
                "observer": "imatrix_mse",
                "observer_kwargs": {},
                "scale_dtype": "torch.float8_e4m3fn",
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
                "zp_dtype": None,
            },
        },
    },
}

MOE_CONFIG = {
    "format": "mixed-precision",
    "global_compression_ratio": None,
    "ignore": [],
    "config_groups": {
        "group_0": {
            "format": "float-quantized",
            "input_activations": FP8_INPUT_ACTIVATIONS,
            "output_activations": None,
            "targets": [
                r"re:.*self_attn\.(q|k|v|o)_proj$",
                r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                r"re:.*lm_head",
            ],
            "weights": FP8_WEIGHTS,
        },
        "group_1": {
            "format": "nvfp4-pack-quantized",
            "input_activations": NVFP4_INPUT_ACTIVATIONS,
            "output_activations": None,
            "targets": [
                r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$",
                r"re:.*shared_expert\.(gate|up|down)_proj$",
            ],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": 16,
                "num_bits": 4,
                "observer": "memoryless_minmax",
                "observer_kwargs": {},
                "scale_dtype": "torch.float8_e4m3fn",
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
                "zp_dtype": None,
            },
        },
    },
}


@pytest.mark.parametrize("config", [DENSE_CONFIG, MOE_CONFIG])
def test_parser_preserves_each_group_format_and_activations(config):
    target_scheme_map = CompressedTensorsConfig._quantization_scheme_map_from_config(
        deepcopy(config)
    )
    fp8_target = config["config_groups"]["group_0"]["targets"][0]
    nvfp4_target = config["config_groups"]["group_1"]["targets"][0]

    assert target_scheme_map[fp8_target]["format"] == "float-quantized"
    assert target_scheme_map[fp8_target]["input_activations"] is not None
    assert target_scheme_map[nvfp4_target]["format"] == "nvfp4-pack-quantized"
    assert target_scheme_map[nvfp4_target]["input_activations"] is not None


def test_dense_dispatch_uses_matched_group_format():
    quant_config = CompressedTensorsConfig.from_config(deepcopy(DENSE_CONFIG))
    quant_config._check_scheme_supported = lambda *args, **kwargs: True
    layer = torch.nn.Linear(4, 4)

    fp8_scheme = quant_config.get_linear_scheme(
        layer, "model.language_model.layers.1.self_attn.q_proj"
    )
    nvfp4_scheme = quant_config.get_linear_scheme(
        layer, "model.language_model.layers.1.mlp.gate_proj"
    )

    assert isinstance(fp8_scheme, CompressedTensorsW8A8Fp8)
    assert isinstance(nvfp4_scheme, CompressedTensorsW4A4Fp4)


def test_moe_dispatch_receives_nvfp4_input_quantization(monkeypatch):
    monkeypatch.setattr(
        compressed_tensors_w4a4_nvfp4_moe,
        "is_blackwell_supported",
        lambda: True,
    )
    quant_config = CompressedTensorsConfig.from_config(deepcopy(MOE_CONFIG))
    layer = torch.nn.Module()
    layer_name = "model.language_model.layers.1.mlp.experts"

    matched = quant_config.get_scheme_dict(layer, f"{layer_name}.0.gate_proj")
    assert matched is not None
    assert matched["format"] == "nvfp4-pack-quantized"
    assert matched["input_activations"] is not None

    scheme = quant_config.get_moe_scheme(layer, layer_name)
    assert isinstance(scheme, CompressedTensorsW4A4Nvfp4MoE)

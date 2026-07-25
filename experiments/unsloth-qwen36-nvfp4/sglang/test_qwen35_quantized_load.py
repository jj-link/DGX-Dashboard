"""CPU-safe sentinel loads for Qwen3.5 mixed FP8/NVFP4 checkpoints."""

from copy import deepcopy
from types import SimpleNamespace

import torch

from test_compressed_tensors_mixed_precision import DENSE_CONFIG
from sglang.srt.layers import vocab_parallel_embedding
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4_moe,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
    CompressedTensorsW4A4Nvfp4MoE,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5GatedDeltaNet


LINEAR_ATTN_PREFIX = "model.language_model.layers.0.linear_attn"


def _dense_quant_config():
    config = deepcopy(DENSE_CONFIG)
    # Exact representative ignore shape from both pinned checkpoints. The
    # parent is metadata-only; the explicit norm/b/a leaves are unquantized.
    config["ignore"] = [
        LINEAR_ATTN_PREFIX,
        f"{LINEAR_ATTN_PREFIX}.norm",
        f"{LINEAR_ATTN_PREFIX}.in_proj_b",
        f"{LINEAR_ATTN_PREFIX}.in_proj_a",
    ]
    quant_config = CompressedTensorsConfig.from_config(config)
    quant_config.packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    quant_config._check_scheme_supported = lambda *args, **kwargs: True
    return quant_config


def _make_load_harness(qkvz, out_proj, lm_head):
    harness = torch.nn.Module()
    harness.model = torch.nn.Module()
    harness.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    harness.model.layers[0].linear_attn = torch.nn.Module()
    harness.model.layers[0].linear_attn.in_proj_qkvz = qkvz
    harness.model.layers[0].linear_attn.out_proj = out_proj
    harness.lm_head = lm_head
    harness.config = SimpleNamespace(tie_word_embeddings=False)
    harness.pp_group = SimpleNamespace(is_last_rank=True)
    harness.start_layer = 0
    harness.end_layer = 1
    return harness


def test_qwen35_fp8_scale_sentinels_load_exactly(monkeypatch):
    quant_config = _dense_quant_config()
    out_proj = RowParallelLinear(
        8,
        8,
        bias=False,
        input_is_parallel=True,
        reduce_results=False,
        quant_config=quant_config,
        prefix=f"{LINEAR_ATTN_PREFIX}.out_proj",
        tp_rank=0,
        tp_size=1,
    )
    qkvz = MergedColumnParallelLinear(
        8,
        [2, 2, 2, 4],
        bias=False,
        quant_config=quant_config,
        prefix=f"{LINEAR_ATTN_PREFIX}.in_proj_qkvz",
        tp_rank=0,
        tp_size=1,
    )
    Qwen3_5GatedDeltaNet._bind_packed_weight_loaders(
        object.__new__(Qwen3_5GatedDeltaNet), qkvz
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_parallel",
        lambda: SimpleNamespace(tp_rank=0, tp_size=1),
    )
    lm_head = ParallelLMHead(
        16,
        8,
        bias=False,
        quant_config=quant_config,
        prefix="lm_head",
        padding_size=1,
    )

    assert hasattr(qkvz, "weight_scale")
    assert hasattr(out_proj, "weight_scale")
    assert hasattr(lm_head, "weight_scale")

    qkv_scale = torch.tensor([[11.0], [12.0], [21.0], [22.0], [31.0], [32.0]])
    z_scale = torch.tensor([[41.0], [42.0], [43.0], [44.0]])
    out_scale = torch.arange(51.0, 59.0).reshape(8, 1)
    lm_head_scale = torch.arange(61.0, 77.0).reshape(16, 1)

    harness = _make_load_harness(qkvz, out_proj, lm_head)
    loaded = Qwen3_5ForCausalLM.load_weights(
        harness,
        [
            (
                "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale",
                qkv_scale,
            ),
            (
                "model.language_model.layers.0.linear_attn.in_proj_z.weight_scale",
                z_scale,
            ),
            (
                "model.language_model.layers.0.linear_attn.out_proj.weight_scale",
                out_scale,
            ),
            ("lm_head.weight_scale", lm_head_scale),
        ],
    )

    assert "model.layers.0.linear_attn.in_proj_qkvz.weight_scale" in loaded
    assert "model.layers.0.linear_attn.out_proj.weight_scale" in loaded
    assert "lm_head.weight_scale" in loaded
    torch.testing.assert_close(
        qkvz.weight_scale,
        torch.cat((qkv_scale, z_scale), dim=0),
    )
    torch.testing.assert_close(out_proj.weight_scale, out_scale)
    torch.testing.assert_close(lm_head.weight_scale, lm_head_scale)


def test_qwen35_moe_registers_all_nvfp4_expert_parameter_families(monkeypatch):
    monkeypatch.setattr(
        compressed_tensors_w4a4_nvfp4_moe,
        "is_blackwell_supported",
        lambda: True,
    )
    scheme = CompressedTensorsW4A4Nvfp4MoE()
    experts = torch.nn.Module()
    scheme.create_weights(
        experts,
        num_experts=2,
        hidden_size=32,
        intermediate_size_per_partition=32,
        params_dtype=torch.bfloat16,
    )

    families = (
        "weight_packed",
        "weight_scale",
        "weight_global_scale",
        "input_global_scale",
    )
    registered = set(dict(experts.named_parameters()))
    assert registered == {
        f"{projection}_{family}"
        for projection in ("w13", "w2")
        for family in families
    }

    mappings = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=2,
    )
    for runtime_prefix, checkpoint_prefix, _, _ in mappings:
        projection = "w13" if runtime_prefix == "experts.w13_" else "w2"
        for family in families:
            checkpoint_name = f"{checkpoint_prefix}{family}"
            runtime_name = checkpoint_name.replace(checkpoint_prefix, runtime_prefix)
            assert runtime_name == f"experts.{projection}_{family}"

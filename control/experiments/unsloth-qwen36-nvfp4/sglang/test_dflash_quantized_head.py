"""CPU-safe DFlash greedy-head tests for an FP8 ParallelLMHead."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.speculative import dflash_worker_v2
from sglang.srt.speculative.dflash_worker_v2 import (
    DFlashWorkerV2,
    _DflashDraftSampler,
)


class SentinelFp8LinearMethod:
    def __init__(self):
        self.call_shapes = []

    def apply(self, layer, hidden_states, bias=None):
        self.call_shapes.append(tuple(hidden_states.shape))
        # CompressedTensorsW8A8Fp8 stores processed channel weights as [K, N].
        dequantized = layer.weight.float() * layer.weight_scale.view(1, -1)
        logits = hidden_states.float() @ dequantized
        if bias is not None:
            logits = logits + bias
        return logits


def _fp8_parallel_lm_head():
    head = object.__new__(ParallelLMHead)
    torch.nn.Module.__init__(head)
    # Runtime orientation after channel-FP8 post-load processing: [hidden, vocab].
    processed_weight = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.float8_e4m3fn,
    )
    # Scale makes token 0 beat raw-storage token 1. Token 7 would win globally,
    # so num_org truncation must happen after the quantized logits computation.
    weight_scale = torch.tensor(
        [[1.0], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [100.0]],
        dtype=torch.float32,
    )
    head.register_parameter(
        "weight", torch.nn.Parameter(processed_weight, requires_grad=False)
    )
    head.register_parameter(
        "weight_scale", torch.nn.Parameter(weight_scale, requires_grad=False)
    )
    head.quant_method = SentinelFp8LinearMethod()
    head.shard_indices = SimpleNamespace(
        num_org_elements=7,
        num_org_elements_padded=8,
        num_added_elements=0,
        org_vocab_start_index=5,
        added_vocab_start_index=12,
    )
    return head


def _hidden_states(num_tokens):
    hidden = torch.zeros((num_tokens, 4), dtype=torch.bfloat16)
    hidden[:, 0] = torch.arange(1, num_tokens + 1, dtype=torch.bfloat16)
    hidden[:, 1:] = 0.25
    return hidden


@pytest.mark.parametrize("block_size", [2, 8, 12, 16])
def test_dflash_graph_and_eager_heads_use_quantized_parallel_lm_head(
    monkeypatch, block_size
):
    num_tokens = block_size - 1
    lm_head = _fp8_parallel_lm_head()
    hidden_states = _hidden_states(num_tokens)
    expected = torch.full((num_tokens,), 5, dtype=torch.long)

    unscaled_logits = hidden_states.float() @ lm_head.weight.float()
    assert torch.all(torch.argmax(unscaled_logits[:, :7], dim=-1) == 1)
    full_scaled_logits = lm_head.quant_method.apply(lm_head, hidden_states)
    assert torch.all(torch.argmax(full_scaled_logits, dim=-1) == 7)
    lm_head.quant_method.call_shapes.clear()

    sampler = _DflashDraftSampler(
        lm_head=lm_head,
        block_size=block_size,
        num_org=7,
        org_vocab_start=5,
        max_bs=1,
    )
    captured_hidden = torch.cat(
        (torch.zeros((1, 4), dtype=torch.bfloat16), hidden_states), dim=0
    )
    sampler(captured_hidden)
    torch.testing.assert_close(sampler.out[:num_tokens], expected)
    assert lm_head.quant_method.call_shapes == [(num_tokens, 4)]

    worker = object.__new__(DFlashWorkerV2)
    worker._draft_greedy_local_cap = 0
    worker._draft_greedy_local_max_buf = None
    worker._draft_greedy_local_arg_buf = None
    worker._draft_greedy_gather_cap = 0
    worker._draft_greedy_gathered_max_buf = None
    worker._draft_greedy_gathered_ids_buf = None
    lm_head.quant_method.call_shapes.clear()
    monkeypatch.setattr(
        dflash_worker_v2,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )

    eager_tokens = DFlashWorkerV2._greedy_sample_from_vocab_parallel_head(
        worker,
        hidden_states=hidden_states,
        lm_head=lm_head,
    )
    torch.testing.assert_close(eager_tokens, expected)
    assert lm_head.quant_method.call_shapes == [(num_tokens, 4)]

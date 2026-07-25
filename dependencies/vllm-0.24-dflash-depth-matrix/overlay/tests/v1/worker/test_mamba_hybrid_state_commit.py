from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


def make_state(*, block_size: int, depth: int):
    conv_state = torch.zeros((32, 1), dtype=torch.float32)
    ssm_state = torch.zeros((32, 1), dtype=torch.float32)
    layer = SimpleNamespace(kv_cache=(conv_state, ssm_state))
    vllm_config = SimpleNamespace(
        num_speculative_tokens=depth,
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        compilation_config=SimpleNamespace(
            static_forward_context={"layer": layer}
        ),
    )
    spec = MambaSpec(
        block_size=block_size,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=depth,
    )
    group = KVCacheGroupSpec(layer_names=["layer"], kv_cache_spec=spec)
    cache_config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[group],
    )
    block_table = torch.arange(1, 33, dtype=torch.int32).unsqueeze(0)

    state = object.__new__(MambaHybridModelState)
    state.is_spec_decode_align_mode = True
    state.vllm_config = vllm_config
    state.last_kv_cache_config = cache_config
    state.last_block_tables = (block_table,)
    return state, layer, block_table


def run_commit(
    state: MambaHybridModelState,
    *,
    seq_len: int,
    num_computed_after: int,
    accepted: int,
    depth: int,
) -> None:
    input_batch = SimpleNamespace(
        num_draft_tokens=depth,
        idx_mapping=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
    )
    state._copy_ssm_staging_to_committed(
        input_batch,
        torch.tensor([accepted], dtype=torch.int32),
        torch.tensor([num_computed_after], dtype=torch.int32),
    )


@pytest.mark.parametrize("accepted", [2, 3, 5])
def test_copies_last_accepted_state_to_persistent_blocks(accepted: int) -> None:
    block_size = 8
    depth = 4
    num_computed_before = 18
    num_computed_after = num_computed_before + accepted
    seq_len = num_computed_before + depth + 1
    state, layer, block_table = make_state(block_size=block_size, depth=depth)

    window_start = (seq_len - 1) // block_size
    source_logical = window_start + accepted - 1
    source_physical = int(block_table[0, source_logical])
    ssm_destination = int(
        block_table[0, (num_computed_after - 1) // block_size]
    )
    conv_destination = int(
        block_table[0, (num_computed_after + depth) // block_size]
    )
    layer.kv_cache[0][source_physical] = 101
    layer.kv_cache[1][source_physical] = 202

    run_commit(
        state,
        seq_len=seq_len,
        num_computed_after=num_computed_after,
        accepted=accepted,
        depth=depth,
    )

    assert layer.kv_cache[0][conv_destination].item() == 101
    assert layer.kv_cache[1][ssm_destination].item() == 202


@pytest.mark.parametrize(
    ("block_size", "depth", "num_computed_before"),
    [(8, 4, 23), (1680, 11, 5039)],
)
def test_boundary_copy_preserves_boundary_state_before_aliasing_write(
    block_size: int, depth: int, num_computed_before: int
) -> None:
    accepted = 2
    num_computed_after = num_computed_before + accepted
    seq_len = num_computed_before + depth + 1
    state, layer, block_table = make_state(block_size=block_size, depth=depth)

    window_start = (seq_len - 1) // block_size
    boundary_source_physical = int(block_table[0, window_start])
    final_source_physical = int(block_table[0, window_start + accepted - 1])
    completed_block_physical = int(
        block_table[0, (num_computed_before - 1) // block_size]
    )
    committed_block_physical = int(
        block_table[0, (num_computed_after - 1) // block_size]
    )
    assert boundary_source_physical == committed_block_physical

    layer.kv_cache[0][boundary_source_physical] = 11
    layer.kv_cache[1][boundary_source_physical] = 22
    layer.kv_cache[0][final_source_physical] = 33
    layer.kv_cache[1][final_source_physical] = 44

    run_commit(
        state,
        seq_len=seq_len,
        num_computed_after=num_computed_after,
        accepted=accepted,
        depth=depth,
    )

    assert layer.kv_cache[0][completed_block_physical].item() == 11
    assert layer.kv_cache[1][completed_block_physical].item() == 22
    assert layer.kv_cache[0][committed_block_physical].item() == 33
    assert layer.kv_cache[1][committed_block_physical].item() == 44


def test_does_not_copy_without_an_accepted_draft() -> None:
    state, layer, _ = make_state(block_size=8, depth=4)
    layer.kv_cache[0][3] = 11
    layer.kv_cache[1][3] = 22

    run_commit(
        state,
        seq_len=23,
        num_computed_after=19,
        accepted=1,
        depth=4,
    )

    assert layer.kv_cache[0][3].item() == 11
    assert layer.kv_cache[1][3].item() == 22

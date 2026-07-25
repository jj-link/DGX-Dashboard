from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

DEPTH_BLOCK_SIZES = (
    (2, 1600),
    (4, 1616),
    (8, 1648),
    (9, 1664),
    (10, 1680),
    (11, 1680),
    (12, 1696),
    (15, 1728),
    (16, 1728),
)


def make_state(*, block_size: int, depth: int):
    conv_state = torch.zeros((64, 1), dtype=torch.float32)
    ssm_state = torch.zeros((64, 1), dtype=torch.float32)
    layer = SimpleNamespace(kv_cache=(conv_state, ssm_state))
    state = object.__new__(MambaHybridModelState)
    state.is_spec_decode_align_mode = True
    state.vllm_config = SimpleNamespace(
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
    state.last_kv_cache_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[group],
    )
    block_table = torch.arange(1, 65, dtype=torch.int32).unsqueeze(0)
    state.last_block_tables = (block_table,)
    state.last_num_reqs = 1
    state.last_num_draft_tokens = depth
    return state, layer, block_table


def run_commit(
    state: MambaHybridModelState,
    *,
    seq_len: int,
    accepted: int,
    depth: int,
) -> None:
    state.last_seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    state.last_query_start_loc = torch.tensor([0, depth + 1], dtype=torch.int32)
    state._copy_staging_to_committed(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([accepted], dtype=torch.int32),
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
    source_physical = int(block_table[0, window_start + accepted - 1])
    ssm_destination = int(
        block_table[0, (num_computed_after - 1) // block_size]
    )
    conv_destination = int(
        block_table[0, (num_computed_after + depth) // block_size]
    )
    layer.kv_cache[0][source_physical] = 101
    layer.kv_cache[1][source_physical] = 202

    run_commit(state, seq_len=seq_len, accepted=accepted, depth=depth)

    assert layer.kv_cache[0][conv_destination].item() == 101
    assert layer.kv_cache[1][ssm_destination].item() == 202


@pytest.mark.parametrize(("depth", "block_size"), DEPTH_BLOCK_SIZES)
def test_boundary_copy_preserves_state_before_aliasing_write(
    depth: int, block_size: int
) -> None:
    accepted = 2
    num_computed_before = 3 * block_size - 1
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

    run_commit(state, seq_len=seq_len, accepted=accepted, depth=depth)

    assert layer.kv_cache[0][completed_block_physical].item() == 11
    assert layer.kv_cache[1][completed_block_physical].item() == 22
    assert layer.kv_cache[0][committed_block_physical].item() == 33
    assert layer.kv_cache[1][committed_block_physical].item() == 44


def test_does_not_copy_without_an_accepted_draft() -> None:
    state, layer, _ = make_state(block_size=8, depth=4)
    layer.kv_cache[0][3] = 11
    layer.kv_cache[1][3] = 22

    run_commit(state, seq_len=23, accepted=1, depth=4)

    assert layer.kv_cache[0][3].item() == 11
    assert layer.kv_cache[1][3].item() == 22

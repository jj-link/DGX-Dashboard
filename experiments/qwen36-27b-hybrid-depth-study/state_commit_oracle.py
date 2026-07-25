#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_DEPTHS = (0, 2, 4, 8, 9, 10, 11, 12, 15, 16)
DEFAULT_BLOCK_SIZES = {
    0: 1568,
    2: 1600,
    4: 1616,
    8: 1648,
    9: 1664,
    10: 1680,
    11: 1680,
    12: 1696,
    15: 1728,
    16: 1728,
}


@dataclass(frozen=True)
class CommitPlan:
    draft_depth: int
    block_size: int
    offset_before: int
    num_computed_before: int
    num_scheduled_tokens: int
    num_accepted_tokens: int
    num_computed_after: int
    forward_window_start: int
    forward_window_ahead_of_committed: bool
    accepted_source_column: int
    accepted_source_logical: int
    ssm_destination_logical: int
    conv_destination_logical: int
    crossed_boundary: bool
    copy_required: bool
    boundary_copy_required: bool
    boundary_source_column: int | None
    boundary_destination_logical: int | None
    next_window_start: int
    scheduled_window_shifted: bool
    expected_state_marker: str
    legacy_persistent_state_marker: str
    oracle_persistent_state_marker: str
    legacy_invariant_passed: bool
    oracle_invariant_passed: bool


def _block_for_last_token(num_tokens: int, block_size: int) -> int:
    return max(num_tokens - 1, 0) // block_size


def derive_commit_plan(
    *,
    draft_depth: int,
    block_size: int,
    offset_before: int,
    num_accepted_tokens: int,
    base_block: int = 2,
) -> CommitPlan:
    if draft_depth < 0:
        raise ValueError("draft_depth must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0 <= offset_before < block_size:
        raise ValueError("offset_before must identify a token within one block")
    max_accepted = draft_depth + 1 if draft_depth else 1
    if not 1 <= num_accepted_tokens <= max_accepted:
        raise ValueError(
            f"num_accepted_tokens must be in [1, {max_accepted}] for depth {draft_depth}"
        )

    num_computed_before = base_block * block_size + offset_before + 1
    num_scheduled_tokens = draft_depth + 1 if draft_depth else 1
    num_computed_after = num_computed_before + num_accepted_tokens

    # mamba_get_block_table_tensor selects its align-mode staging window from
    # the scheduled sequence length, not only the committed sequence length.
    forward_seq_len = num_computed_before + num_scheduled_tokens
    forward_window_start = _block_for_last_token(forward_seq_len, block_size)
    accepted_source_column = num_accepted_tokens - 1
    accepted_source_logical = forward_window_start + accepted_source_column

    previous_committed_block = _block_for_last_token(
        num_computed_before, block_size
    )
    forward_window_ahead_of_committed = (
        forward_window_start > previous_committed_block
    )
    ssm_destination_logical = _block_for_last_token(num_computed_after, block_size)
    # Mirrors MambaHybridModelState._copy_ssm_staging_to_committed: the
    # convolution working block accounts for the full configured draft depth.
    conv_destination_logical = max(
        num_computed_after + draft_depth, 0
    ) // block_size

    crossed_boundary = ssm_destination_logical > previous_committed_block
    copy_required = draft_depth > 0 and num_accepted_tokens > 1
    boundary_copy_required = copy_required and crossed_boundary

    boundary_source_column = None
    boundary_destination_logical = None
    if boundary_copy_required:
        boundary_last_token = (previous_committed_block + 1) * block_size - 1
        boundary_source_column = min(
            max(boundary_last_token - num_computed_before, 0), draft_depth
        )
        boundary_destination_logical = previous_committed_block

    next_seq_len = num_computed_after + num_scheduled_tokens
    next_window_start = _block_for_last_token(next_seq_len, block_size)
    scheduled_window_shifted = next_window_start != forward_window_start

    expected_state_marker = f"h{num_accepted_tokens}"
    # The unpatched path records the accepted count but does not copy a state
    # produced in staging column a-1 back into the persistent align-mode state.
    legacy_persistent_state_marker = (
        expected_state_marker if not copy_required else "h0"
    )
    oracle_persistent_state_marker = expected_state_marker

    return CommitPlan(
        draft_depth=draft_depth,
        block_size=block_size,
        offset_before=offset_before,
        num_computed_before=num_computed_before,
        num_scheduled_tokens=num_scheduled_tokens,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_after=num_computed_after,
        forward_window_start=forward_window_start,
        forward_window_ahead_of_committed=forward_window_ahead_of_committed,
        accepted_source_column=accepted_source_column,
        accepted_source_logical=accepted_source_logical,
        ssm_destination_logical=ssm_destination_logical,
        conv_destination_logical=conv_destination_logical,
        crossed_boundary=crossed_boundary,
        copy_required=copy_required,
        boundary_copy_required=boundary_copy_required,
        boundary_source_column=boundary_source_column,
        boundary_destination_logical=boundary_destination_logical,
        next_window_start=next_window_start,
        scheduled_window_shifted=scheduled_window_shifted,
        expected_state_marker=expected_state_marker,
        legacy_persistent_state_marker=legacy_persistent_state_marker,
        oracle_persistent_state_marker=oracle_persistent_state_marker,
        legacy_invariant_passed=(legacy_persistent_state_marker == expected_state_marker),
        oracle_invariant_passed=True,
    )


def contiguous_ranges(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    ranges: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])
    return ranges


def evaluate_matrix(
    *, depths: tuple[int, ...], block_size_override: int | None = None
) -> dict[str, object]:
    block_sizes = {
        depth: (
            block_size_override
            if block_size_override is not None
            else DEFAULT_BLOCK_SIZES[depth]
        )
        for depth in depths
    }
    summaries: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    total_cases = 0
    total_legacy_failures = 0
    total_boundary_cases = 0

    for depth in depths:
        block_size = block_sizes[depth]
        accepted_limit = depth + 1 if depth else 1
        depth_cases = 0
        depth_legacy_failures = 0
        depth_boundary_cases = 0
        by_accepted: list[dict[str, object]] = []

        for accepted in range(1, accepted_limit + 1):
            copy_offsets: list[int] = []
            boundary_offsets: list[int] = []
            shifted_offsets: list[int] = []
            forward_ahead_offsets: list[int] = []
            legacy_failure_offsets: list[int] = []
            for offset in range(block_size):
                plan = derive_commit_plan(
                    draft_depth=depth,
                    block_size=block_size,
                    offset_before=offset,
                    num_accepted_tokens=accepted,
                )
                depth_cases += 1
                if plan.copy_required:
                    copy_offsets.append(offset)
                if plan.boundary_copy_required:
                    boundary_offsets.append(offset)
                    depth_boundary_cases += 1
                if plan.scheduled_window_shifted:
                    shifted_offsets.append(offset)
                if plan.forward_window_ahead_of_committed:
                    forward_ahead_offsets.append(offset)
                if not plan.legacy_invariant_passed:
                    legacy_failure_offsets.append(offset)
                    depth_legacy_failures += 1

                if (
                    offset in {0, block_size - accepted, block_size - 1}
                    and len(samples) < 256
                ):
                    samples.append(asdict(plan))

            by_accepted.append(
                {
                    "num_accepted_tokens": accepted,
                    "copy_required_offset_ranges": contiguous_ranges(copy_offsets),
                    "boundary_copy_offset_ranges": contiguous_ranges(boundary_offsets),
                    "scheduled_window_shift_offset_ranges": contiguous_ranges(
                        shifted_offsets
                    ),
                    "forward_window_ahead_offset_ranges": contiguous_ranges(
                        forward_ahead_offsets
                    ),
                    "legacy_failure_offset_ranges": contiguous_ranges(
                        legacy_failure_offsets
                    ),
                }
            )

        total_cases += depth_cases
        total_legacy_failures += depth_legacy_failures
        total_boundary_cases += depth_boundary_cases
        summaries.append(
            {
                "draft_depth": depth,
                "accepted_counts_exercised": accepted_limit,
                "offsets_exercised": block_size,
                "cases": depth_cases,
                "legacy_invariant_failures": depth_legacy_failures,
                "boundary_copy_cases": depth_boundary_cases,
                "by_accepted_count": by_accepted,
            }
        )

    return {
        "schema_version": 1,
        "block_size_override": block_size_override,
        "block_sizes": {str(depth): block_sizes[depth] for depth in depths},
        "depths": list(depths),
        "total_cases": total_cases,
        "total_legacy_invariant_failures": total_legacy_failures,
        "total_boundary_copy_cases": total_boundary_cases,
        "candidate_predicates": {
            "copy_required": "draft_depth > 0 and num_accepted_tokens > 1",
            "boundary_copy_required": (
                "copy_required and floor((num_computed_after - 1) / block_size) "
                "> floor((num_computed_before - 1) / block_size)"
            ),
            "forward_window_ahead_of_committed": (
                "floor((num_computed_before + draft_depth) / block_size) > "
                "floor((num_computed_before - 1) / block_size)"
            ),
            "scheduled_window_shifted": (
                "floor((num_computed_after + draft_depth) / block_size) != "
                "floor((num_computed_before + draft_depth) / block_size)"
            ),
        },
        "summaries": summaries,
        "samples": samples,
    }


def self_test() -> None:
    plain = derive_commit_plan(
        draft_depth=0,
        block_size=8,
        offset_before=7,
        num_accepted_tokens=1,
    )
    assert not plain.copy_required
    assert plain.legacy_invariant_passed

    no_boundary = derive_commit_plan(
        draft_depth=4,
        block_size=8,
        offset_before=1,
        num_accepted_tokens=3,
    )
    assert no_boundary.copy_required
    assert not no_boundary.crossed_boundary
    assert not no_boundary.legacy_invariant_passed
    assert no_boundary.oracle_invariant_passed

    boundary = derive_commit_plan(
        draft_depth=4,
        block_size=8,
        offset_before=6,
        num_accepted_tokens=2,
    )
    assert boundary.boundary_copy_required
    assert boundary.boundary_source_column == 0
    assert boundary.ssm_destination_logical == boundary.boundary_destination_logical + 1

    matrix = evaluate_matrix(depths=(0, 2, 4), block_size_override=8)
    assert matrix["total_cases"] == 8 * (1 + 3 + 5)
    assert matrix["total_legacy_invariant_failures"] == 8 * (2 + 4)
    assert matrix["total_boundary_copy_cases"] == (2 + 3) + (2 + 3 + 4 + 5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEPTHS),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    result = evaluate_matrix(
        depths=tuple(args.depths), block_size_override=args.block_size
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


def parse_sse(path: Path) -> list[str]:
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        choices = event.get("choices") or []
        if not choices:
            continue
        content = choices[0].get("delta", {}).get("content")
        if content:
            chunks.append(content)
    return chunks


def common_prefix_length(texts: list[str]) -> int:
    for index, characters in enumerate(zip(*texts)):
        if len(set(characters)) != 1:
            return index
    return min(map(len, texts))


BLOCK_SIZE_RE = re.compile(
    r"Setting attention block size to (?P<tokens>\d+) tokens"
)


def parse_block_size(server_log: Path) -> int:
    match = BLOCK_SIZE_RE.search(server_log.read_text())
    if match is None:
        raise ValueError(f"attention block size not found in {server_log}")
    return int(match.group("tokens"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "model": str(args.model),
        "block_size_override": args.block_size,
        "depths": {},
    }
    for depth_dir in sorted(args.sweep.glob("depth-*"), key=lambda p: int(p.name[6:])):
        result_path = depth_dir / "result.json"
        if not result_path.exists():
            continue
        depth = int(depth_dir.name[6:])
        result = json.loads(result_path.read_text())
        block_size = args.block_size or parse_block_size(depth_dir / "server.log")
        texts = [
            (depth_dir / "outputs" / f"agent-{repetition}.txt").read_text()
            for repetition in range(1, 4)
        ]
        common_chars = common_prefix_length(texts)
        repetitions: list[dict[str, object]] = []
        for repetition, text in enumerate(texts, 1):
            chunks = parse_sse(
                depth_dir / "raw" / f"agent-{repetition}.response.sse"
            )
            characters_before_step = 0
            divergent_step = None
            divergent_chunk = None
            for step, chunk in enumerate(chunks, 1):
                if characters_before_step + len(chunk) > common_chars:
                    divergent_step = step
                    divergent_chunk = chunk
                    break
                characters_before_step += len(chunk)
            prefix_before_step = text[:characters_before_step]
            completion_tokens_before_step = len(
                tokenizer.encode(prefix_before_step, add_special_tokens=False)
            )
            prompt_tokens = result["requests"][repetition - 1]["usage"][
                "prompt_tokens"
            ]
            encoding = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            output_tokens = encoding["input_ids"]
            token_end_offsets = [end for _, end in encoding["offset_mapping"]]
            completion_tokens_to_first_boundary = max(
                block_size - prompt_tokens, 0
            )
            boundary_prefix = tokenizer.decode(
                output_tokens[:completion_tokens_to_first_boundary],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            boundary_character = len(boundary_prefix)
            boundary_round_trip_exact = text.startswith(boundary_prefix)
            boundary_stream_step = None
            boundary_step_tokens_before = None
            boundary_step_tokens_emitted = None
            boundary_step_accepted_drafts = None
            first_multitoken_step_after_boundary = None
            stream_character_end = 0
            stream_tokens_before = 0
            for stream_step, chunk in enumerate(chunks, 1):
                stream_character_end += len(chunk)
                stream_tokens_after = bisect_right(
                    token_end_offsets, stream_character_end
                )
                emitted = stream_tokens_after - stream_tokens_before
                if (
                    boundary_stream_step is None
                    and stream_tokens_after >= completion_tokens_to_first_boundary
                ):
                    boundary_stream_step = stream_step
                    boundary_step_tokens_before = stream_tokens_before
                    boundary_step_tokens_emitted = emitted
                    boundary_step_accepted_drafts = max(emitted - 1, 0)
                if (
                    first_multitoken_step_after_boundary is None
                    and prompt_tokens + stream_tokens_before >= block_size
                    and emitted > 1
                ):
                    first_multitoken_step_after_boundary = {
                        "stream_step": stream_step,
                        "num_computed_before_step": (
                            prompt_tokens + stream_tokens_before
                        ),
                        "block_offset_before_step": (
                            prompt_tokens + stream_tokens_before
                        )
                        % block_size,
                        "tokens_emitted": emitted,
                        "accepted_drafts": emitted - 1,
                        "chunk": chunk,
                    }
                stream_tokens_before = stream_tokens_after
            num_computed_before_step = prompt_tokens + completion_tokens_before_step
            repetitions.append(
                {
                    "repetition": repetition,
                    "divergent_step": divergent_step,
                    "divergent_chunk": divergent_chunk,
                    "characters_before_step": characters_before_step,
                    "completion_tokens_before_step": completion_tokens_before_step,
                    "prompt_tokens": prompt_tokens,
                    "first_boundary_completion_token": (
                        completion_tokens_to_first_boundary
                    ),
                    "first_boundary_character": boundary_character,
                    "first_boundary_line": text.count("\n", 0, boundary_character) + 1,
                    "first_boundary_round_trip_exact": boundary_round_trip_exact,
                    "first_boundary_context": text[
                        max(boundary_character - 80, 0) : boundary_character + 160
                    ],
                    "first_boundary_stream_step": boundary_stream_step,
                    "first_boundary_step_tokens_before": (
                        boundary_step_tokens_before
                    ),
                    "first_boundary_step_tokens_emitted": (
                        boundary_step_tokens_emitted
                    ),
                    "first_boundary_step_accepted_drafts": (
                        boundary_step_accepted_drafts
                    ),
                    "first_multitoken_step_after_boundary": (
                        first_multitoken_step_after_boundary
                    ),
                    "num_computed_before_step": num_computed_before_step,
                    "block_offset_before_step": num_computed_before_step
                    % block_size,
                    "characters_after_common_prefix": text[
                        common_chars : common_chars + 80
                    ],
                }
            )
        report["depths"][str(depth)] = {
            "block_size": block_size,
            "common_prefix_characters": common_chars,
            "common_prefix_line": texts[0].count("\n", 0, common_chars) + 1,
            "common_prefix_tokens": len(
                tokenizer.encode(texts[0][:common_chars], add_special_tokens=False)
            ),
            "output_token_counts": [
                request["usage"]["completion_tokens"]
                for request in result["requests"]
            ],
            "output_hashes": [
                request["output_sha256"] for request in result["requests"]
            ],
            "repetitions": repetitions,
        }

    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

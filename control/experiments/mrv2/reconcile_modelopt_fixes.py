from __future__ import annotations

import sys
from pathlib import Path


def fatal(message: str) -> None:
    raise SystemExit(f"fatal: {message}")


def require_markers(path: Path, markers: tuple[str, ...], role: str) -> str:
    if not path.is_file():
        fatal(f"missing {role}: {path}")
    text = path.read_text()
    missing = [marker for marker in markers if marker not in text]
    if missing:
        fatal(f"{role} {path} is missing expected marker(s): {missing}")
    return text


if len(sys.argv) != 4:
    fatal("usage: reconcile_modelopt_fixes.py HOST_FIX_ROOT MODERN_SITE_ROOT OUTPUT_ROOT")

host_root = Path(sys.argv[1])
modern_root = Path(sys.argv[2])
output_root = Path(sys.argv[3])
output_root.mkdir(parents=True, exist_ok=True)

modelopt_rel = Path("vllm/model_executor/layers/quantization/modelopt.py")
qwen_rel = Path("vllm/model_executor/models/qwen3_5.py")
parameter_rel = Path("vllm/model_executor/parameter.py")
vocab_rel = Path("vllm/model_executor/layers/vocab_parallel_embedding.py")

# First fingerprint the existing host overlays. This makes a changed or missing
# prerequisite fail instead of silently certifying a different patch set.
require_markers(
    host_root / modelopt_rel,
    (
        "class ModelOptMixedPrecisionConfig",
        "_normalize_mixed_quant_algo",
        "model.language_model.",
        "ParallelLMHead",
    ),
    "host ModelOpt fix",
)
require_markers(
    host_root / qwen_rel,
    ("ParallelLMHead(", "quant_config=vllm_config.quant_config"),
    "host Qwen fix",
)
require_markers(
    host_root / vocab_rel,
    ("self.params_dtype", "loaded_weight.reshape(1)"),
    "host vocabulary-loader fix",
)
require_markers(
    host_root / parameter_rel,
    ("VLLM_NVFP4_MERGED_COLUMN_LOAD_FAIL", "DFLASH_NVFP4_ROW_LOAD_FAIL"),
    "host parameter diagnostics",
)

# Reconcile behavior, not old full-file text. The modern pin has evolved past
# the old overlay and supports W4A16_NVFP4 natively; replaying the legacy diff
# would remove that support. These assertions certify the upstream equivalents
# needed by the exact Qwen3.6 ModelOpt checkpoint.
require_markers(
    modern_root / modelopt_rel,
    (
        '"W4A16_NVFP4"',
        "ModelOptNvFp4W4A16LinearMethod",
        "_quantized_layer_prefix_candidates",
        'if isinstance(layer, (LinearBase, ParallelLMHead))',
        'if prefix.startswith("language_model.model.")',
        'elif prefix.startswith("model.language_model.")',
        '"qkv_proj": ("q_proj", "k_proj", "v_proj")',
        '"gate_up_proj": ("gate_proj", "up_proj")',
    ),
    "modern ModelOpt implementation",
)
require_markers(
    modern_root / qwen_rel,
    ("ParallelLMHead(", "quant_config=self.quant_config"),
    "modern Qwen implementation",
)
require_markers(
    modern_root / vocab_rel,
    (
        "self.params_dtype = params_dtype",
        "loaded_weight.ndim == 0",
        "param.data.numel() == 1",
        "loaded_weight = loaded_weight.reshape(1)",
    ),
    "modern vocabulary loader",
)
require_markers(
    modern_root / parameter_rel,
    (
        "def load_merged_column_weight",
        "def load_row_parallel_weight",
        "loaded_weight = loaded_weight.narrow(",
        "assert self.data.shape == loaded_weight.shape",
    ),
    "modern parameter loader",
)

manifest_rows = (
    (str(modelopt_rel), "superseded-upstream", "native W4A16 plus Qwen prefix/fused lookup"),
    (str(qwen_rel), "superseded-upstream", "quantized ParallelLMHead"),
    (str(vocab_rel), "superseded-upstream", "dtype retention plus scalar scale reshape"),
    (
        str(parameter_rel),
        "diagnostic-only-not-forwarded",
        "host delta only prints before existing exceptions; modern loader behavior retained",
    ),
)
(output_root / "manifest.tsv").write_text(
    "".join("\t".join(row) + "\n" for row in manifest_rows)
)

# Syntax-check every reconciled source file in the installed tree.
import py_compile

for rel in (modelopt_rel, qwen_rel, vocab_rel, parameter_rel):
    try:
        py_compile.compile(str(modern_root / rel), doraise=True)
    except py_compile.PyCompileError as exc:
        fatal(str(exc))

print("reconciled four host Qwen/ModelOpt overlays against native modern implementations")

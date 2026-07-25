#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import sys

MARKER = "[DFLASH-MODELOPT-LMHEAD-PATCH]"
IMPORT_ANCHOR = "import torch\n"
IMPORT_NEW = IMPORT_ANCHOR + "from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod\n"

METHOD_ANCHOR = '''    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        \"\"\"Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        \"\"\"

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype
'''

METHOD_NEW = '''    def _greedy_sample_from_lm_head_apply(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        # [DFLASH-MODELOPT-LMHEAD-PATCH]
        # ModelOpt NVFP4 may expose lm_head.weight as a packed tensor whose
        # last dimension is not hidden_size. The normal SGLang logits path
        # handles that via lm_head.quant_method.apply(...); DFlash must do the
        # same for draft hidden-state sampling.
        quant_method = getattr(lm_head, "quant_method", None)
        if quant_method is None:
            raise RuntimeError("DFLASH quantized LM-head fallback requires lm_head.quant_method.")

        num_tokens = int(hidden_states.shape[0])
        out_tokens = torch.empty((num_tokens,), dtype=torch.long, device=hidden_states.device)
        shard = getattr(lm_head, "shard_indices", None)
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        if shard is None:
            num_org = None
            num_org_padded = None
            num_added = 0
            org_vocab_start = 0
            added_vocab_start = 0
        else:
            num_org = int(shard.num_org_elements)
            num_org_padded = int(shard.num_org_elements_padded)
            num_added = int(shard.num_added_elements)
            org_vocab_start = int(shard.org_vocab_start_index)
            added_vocab_start = int(shard.added_vocab_start_index)

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = hidden_states[start:end]
            logits = quant_method.apply(lm_head, hs, None)
            chunk_len = int(logits.shape[0])

            if shard is None:
                local_max, local_arg = torch.max(logits, dim=-1)
                global_ids = local_arg.to(torch.int64)
            else:
                if num_org and num_org > 0:
                    base_logits = logits[:, :num_org]
                    local_max, local_arg = torch.max(base_logits, dim=-1)
                else:
                    local_max = torch.full((chunk_len,), torch.finfo(logits.dtype).min, dtype=logits.dtype, device=logits.device)
                    local_arg = torch.zeros((chunk_len,), dtype=torch.int64, device=logits.device)

                if num_added > 0:
                    added_slice_start = num_org_padded
                    added_slice_end = num_org_padded + num_added
                    added_logits = logits[:, added_slice_start:added_slice_end]
                    added_max, added_arg = torch.max(added_logits, dim=-1)
                    use_added = added_max > local_max
                    local_max = torch.where(use_added, added_max, local_max)
                    local_arg = torch.where(use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg)

                if num_added == 0:
                    global_ids = local_arg.to(torch.int64) + org_vocab_start
                else:
                    global_ids = torch.empty((chunk_len,), dtype=torch.int64, device=logits.device)
                    is_base = local_arg < num_org
                    global_ids[is_base] = org_vocab_start + local_arg[is_base]
                    global_ids[~is_base] = added_vocab_start + (local_arg[~is_base] - num_org_padded)

            if tp_size == 1:
                out_tokens[start:end].copy_(global_ids.to(torch.long))
                continue

            gathered_max = torch.empty((tp_size * chunk_len,), dtype=local_max.dtype, device=local_max.device)
            gathered_ids = torch.empty((tp_size * chunk_len,), dtype=global_ids.dtype, device=global_ids.device)
            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)
            best_rank = torch.argmax(gathered_max, dim=0).view(1, chunk_len)
            selected_ids = torch.gather(gathered_ids, 0, best_rank).view(-1)
            out_tokens[start:end].copy_(selected_ids.to(torch.long))

        return out_tokens

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        \"\"\"Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        \"\"\"

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        weight = lm_head.weight  # [local_vocab_padded, hidden]
        quant_method = getattr(lm_head, "quant_method", None)
        weight_hidden_dim = int(weight.shape[-1]) if hasattr(weight, "shape") and len(weight.shape) >= 2 else -1
        if (quant_method is not None and not isinstance(quant_method, UnquantizedEmbeddingMethod)) or weight_hidden_dim != int(hidden_states.shape[-1]):
            if self.tp_rank == 0 and not getattr(self, "_logged_modelopt_lmhead_apply", False):
                logger.info(
                    "DFLASH using lm_head.quant_method.apply for draft sampling. hidden_dim=%s, lm_head_weight_shape=%s, quant_method=%s",
                    int(hidden_states.shape[-1]),
                    tuple(weight.shape) if hasattr(weight, "shape") else None,
                    type(quant_method).__name__ if quant_method is not None else None,
                )
                self._logged_modelopt_lmhead_apply = True
            return self._greedy_sample_from_lm_head_apply(hidden_states=hidden_states, lm_head=lm_head, chunk_size=chunk_size)
        weight_dtype = weight.dtype
'''

def locate():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not getattr(spec, "submodule_search_locations", None):
        sys.exit("ERROR: sglang package not found; pass dflash_worker_v2.py path explicitly")
    base = list(spec.submodule_search_locations)[0]
    return os.path.join(base, "srt", "speculative", "dflash_worker_v2.py")

def patch(path):
    path = os.path.abspath(path)
    original = open(path, "r", encoding="utf-8").read()
    if MARKER in original:
        print(f"[patch] already patched: {path}")
        return
    if IMPORT_ANCHOR not in original:
        sys.exit(f"ERROR: import anchor not found in {path}")
    if METHOD_ANCHOR not in original:
        sys.exit(f"ERROR: method anchor not found in {path}; SGLang version drifted")
    backup = path + ".dflash-modelopt-lmhead-bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"[patch] backup written: {backup}")
    patched = original.replace(IMPORT_ANCHOR, IMPORT_NEW, 1).replace(METHOD_ANCHOR, METHOD_NEW, 1)
    try:
        ast.parse(patched)
    except SyntaxError as exc:
        sys.exit(f"ERROR: patched file failed AST parse: {exc}")
    open(path, "w", encoding="utf-8").write(patched)
    print(f"[patch] applied: {path}")

if __name__ == "__main__":
    patch(sys.argv[1] if len(sys.argv) > 1 else locate())

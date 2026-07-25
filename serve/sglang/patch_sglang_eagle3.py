#!/usr/bin/env python3
"""patch_sglang_eagle3.py -- enable EAGLE-3 speculative decoding for the
Qwen3.6-class target (Qwen3_5ForConditionalGeneration) in SGLang.

    python3 patch_sglang_eagle3.py
    python3 patch_sglang_eagle3.py /path/to/sglang/srt/models/qwen3_5.py

This is a verbatim copy of the patch shipped with
`Ex0bit/Qwen3.6-27B-PRISM-EAGLE3`, placed in this repo so the serve wrapper
can apply it idempotently inside each container start.
"""
import ast
import importlib.util
import os
import shutil
import sys

MARKER = "[EAGLE3-PATCH]"

DECODER_ANCHOR = (
    "    def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):\n"
    "        self.layers_to_capture = layers_to_capture\n"
    "        for layer_id in self.layers_to_capture:\n"
    '            setattr(self.layers[layer_id], "_is_layer_to_capture", True)\n'
)
DECODER_NEW = DECODER_ANCHOR + (
    "\n"
    "    def set_eagle3_layers_to_capture(self, layers_to_capture: list[int]):\n"
    f"        # {MARKER} EAGLE-3 aux-hidden capture for the dense Qwen3.5/3.6\n"
    "        # decoder. Same mechanism as DFlash: mark the decoder layers whose\n"
    "        # residual-stream input is captured and fed to the EAGLE-3 drafter.\n"
    "        self.layers_to_capture = layers_to_capture\n"
    "        for layer_id in self.layers_to_capture:\n"
    '            setattr(self.layers[layer_id], "_is_layer_to_capture", True)\n'
)

WRAPPER_ANCHOR = (
    "class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):"
)
WRAPPER_NEW = (
    "    def set_eagle3_layers_to_capture(self, layer_ids: Optional[list[int]] = None):\n"
    f"        # {MARKER} Route EAGLE-3 aux-hidden capture through the dense\n"
    "        # Qwen3_5ForCausalLM decoder's per-layer mechanism. The inherited\n"
    "        # Qwen3VLForConditionalGeneration version sets model.layers_to_capture\n"
    "        # as a plain list, which only the base Qwen3LLMModel.forward reads;\n"
    "        # the Qwen3.5/3.6 decoder captures via per-layer _is_layer_to_capture\n"
    "        # attrs instead. layer_ids are HF-style 'after layer k'; the +1\n"
    "        # converts to SGLang's 'capture before layer k+1' convention.\n"
    "        self.capture_aux_hidden_states = True\n"
    "        if layer_ids is None:\n"
    '            text_cfg = getattr(self.config, "text_config", self.config)\n'
    "            num_layers = text_cfg.num_hidden_layers\n"
    "            offset_ids = [2, num_layers // 2, num_layers - 3]\n"
    "        else:\n"
    "            offset_ids = [val + 1 for val in layer_ids]\n"
    "        self.model.set_eagle3_layers_to_capture(offset_ids)\n"
    "\n"
    "\n"
    + WRAPPER_ANCHOR
)


def locate_qwen3_5():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not getattr(spec, "submodule_search_locations", None):
        sys.exit(
            "ERROR: the 'sglang' package was not found. "
            "Pass the qwen3_5.py path explicitly."
        )
    base = list(spec.submodule_search_locations)[0]
    return os.path.join(base, "srt", "models", "qwen3_5.py")


def patch(path):
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as fh:
        original = fh.read()

    if MARKER in original:
        print(f"[patch] already patched: {path}")
        return

    if DECODER_ANCHOR not in original:
        sys.exit(
            f"ERROR: decoder anchor not found in {path}. "
            "SGLang version likely drifted."
        )
    if WRAPPER_ANCHOR not in original:
        sys.exit(
            f"ERROR: wrapper anchor not found in {path}. "
            "SGLang version likely drifted."
        )

    backup = path + ".eagle3-bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"[patch] backup written: {backup}")

    patched = original.replace(DECODER_ANCHOR, DECODER_NEW, 1)
    patched = patched.replace(WRAPPER_ANCHOR, WRAPPER_NEW, 1)

    try:
        ast.parse(patched)
    except SyntaxError as exc:
        sys.exit(f"ERROR: patched file failed AST parse: {exc}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(patched)
    print(f"[patch] applied: {path}")


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else locate_qwen3_5()
    patch(target)


if __name__ == "__main__":
    main()

from vllm.utils.platform_utils import is_pin_memory_available, is_uva_available
import torch

print({
    "pin_available": is_pin_memory_available(),
    "uva_available": is_uva_available(),
    "cuda_available": torch.cuda.is_available(),
})
try:
    tensor = torch.empty(1024, pin_memory=True)
    print({"pin_allocation": True, "is_pinned": tensor.is_pinned()})
except Exception as exc:
    print({"pin_allocation": False, "error": f"{type(exc).__name__}: {exc}"})

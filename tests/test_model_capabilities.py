"""Contracts for live model capability publication."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from control.runtime.model_capabilities import ModelCapabilitiesMiddleware, load_capabilities


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILE = (
    _REPO_ROOT
    / "control/serve/cluster/vllm/deepseek_ai_deepseek_v4_flash_dspark_tp2/capabilities.json"
)


def test_deepseek_profile_loads_with_runtime_identity() -> None:
    document = load_capabilities(
        _PROFILE,
        model="deepseek-v4-flash-0731",
        context_window=1_048_576,
    )

    assert document["model"] == "deepseek-v4-flash-0731"
    assert document["context_window"] == 1_048_576
    assert document["max_output_tokens"] == 393_216
    assert document["quantization"] == {"weights": "NVFP4"}
    assert document["tools"] == {"supported": True, "parallel": True}
    assert document["reasoning"] == {
        "supported": True,
        "can_disable": True,
        "levels": ["high", "max"],
        "default": "high",
        "request_format": "qwen-chat-template",
        "response_field": "reasoning_content",
    }


def test_profile_rejects_unordered_reasoning_levels(tmp_path: Path) -> None:
    payload = json.loads(_PROFILE.read_text(encoding="utf-8"))
    payload["reasoning"]["levels"] = ["max", "high"]
    path = tmp_path / "capabilities.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical order"):
        load_capabilities(path, model="model", context_window=1)


def test_middleware_publishes_document_and_delegates_other_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DGX_MODEL_CAPABILITIES_PATH", str(_PROFILE))
    monkeypatch.setenv("SERVED", "deepseek-v4-flash-0731")
    monkeypatch.setenv("MAX_MODEL_LEN", "1048576")
    delegated: list[str] = []

    async def app(scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        delegated.append(scope["path"])

    middleware = ModelCapabilitiesMiddleware(app)

    async def request(path: str, method: str = "GET") -> list[dict[str, object]]:
        messages: list[dict[str, object]] = []

        async def receive() -> dict[str, object]:
            return {"type": "http.request"}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        await middleware(
            {"type": "http", "path": path, "method": method},
            receive,
            send,
        )
        return messages

    messages = asyncio.run(request("/v1/model-capabilities"))
    assert messages[0]["status"] == 200
    body = json.loads(messages[1]["body"])
    assert body["model"] == "deepseek-v4-flash-0731"
    assert body["reasoning"]["levels"] == ["high", "max"]
    assert delegated == []

    method_messages = asyncio.run(request("/v1/model-capabilities", "POST"))
    assert method_messages[0]["status"] == 405

    asyncio.run(request("/v1/models"))
    assert delegated == ["/v1/models"]

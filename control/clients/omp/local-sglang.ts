import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

import { loadCapabilityDocument, registerCapabilityProvider } from "./capability-provider";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1";
const baseUrl = (process.env.LOCAL_INFERENCE_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/$/, "");
const capabilities = await loadCapabilityDocument(baseUrl);

export default function localSglangCapabilities(pi: ExtensionAPI): void {
  registerCapabilityProvider(pi, { id: "vllm", label: "local RTX 6000", baseUrl }, capabilities);
}

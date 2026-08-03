import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

import { loadCapabilityDocument, registerCapabilityProvider } from "./capability-provider";

const DEFAULT_BASE_URL = "http://100.92.139.82:8888/v1";
const baseUrl = (process.env.SPARK_CLUSTER_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/$/, "");
const capabilities = await loadCapabilityDocument(baseUrl);

export default function sparkClusterCapabilities(pi: ExtensionAPI): void {
  registerCapabilityProvider(pi, { id: "spark-cluster", label: "Spark cluster", baseUrl }, capabilities);
}

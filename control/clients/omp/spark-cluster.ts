import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

const DEFAULT_BASE_URL = "http://100.92.139.82:8888/v1";
const LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"] as const;

const baseUrl = (process.env.SPARK_CLUSTER_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/$/, "");
let capabilityPayload: unknown = null;
let loadError: string | null = null;
try {
  const response = await fetch(`${baseUrl}/model-capabilities`, {
    headers: { accept: "application/json" },
    signal: AbortSignal.timeout(5_000),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  capabilityPayload = await response.json();
} catch (error) {
  loadError = error instanceof Error ? error.message : String(error);
}

export default function sparkClusterCapabilities(pi: ExtensionAPI): void {
  if (capabilityPayload === null) {
    pi.logger.warn(`spark-cluster capability discovery unavailable: ${loadError}`);
    return;
  }

  const { z } = pi.zod;
  const levelSchema = z.enum(LEVELS);
  const capabilitiesSchema = z
    .object({
      object: z.literal("model.capabilities"),
      schema_version: z.literal(1),
      model: z.string().min(1),
      api: z.literal("openai-completions"),
      input: z.array(z.enum(["text", "image"])).min(1),
      context_window: z.number().int().positive(),
      max_output_tokens: z.number().int().positive(),
      tools: z
        .object({
          supported: z.boolean(),
          parallel: z.boolean(),
        })
        .strict(),
      reasoning: z
        .object({
          supported: z.boolean(),
          can_disable: z.boolean(),
          levels: z.array(levelSchema),
          default: levelSchema.nullable(),
          request_format: z.literal("qwen-chat-template").nullable(),
          response_field: z.enum(["reasoning_content", "reasoning"]).nullable(),
        })
        .strict(),
    })
    .strict()
    .superRefine((document, context) => {
      const published = document.reasoning.levels;
      if (new Set(published).size !== published.length) {
        context.addIssue({ code: "custom", path: ["reasoning", "levels"], message: "levels must be unique" });
      }
      const canonical = LEVELS.filter((level) => published.includes(level));
      if (canonical.some((level, index) => level !== published[index])) {
        context.addIssue({ code: "custom", path: ["reasoning", "levels"], message: "levels must use canonical order" });
      }
      if (document.reasoning.supported !== (published.length > 0)) {
        context.addIssue({ code: "custom", path: ["reasoning", "supported"], message: "support must agree with levels" });
      }
      if (document.reasoning.default !== null && !published.includes(document.reasoning.default)) {
        context.addIssue({ code: "custom", path: ["reasoning", "default"], message: "default must be published" });
      }
      if (document.tools.parallel && !document.tools.supported) {
        context.addIssue({ code: "custom", path: ["tools", "parallel"], message: "parallel tools require tool support" });
      }
    });

  const parsed = capabilitiesSchema.safeParse(capabilityPayload);
  if (!parsed.success) {
    pi.logger.error(`spark-cluster capability discovery rejected: ${z.prettifyError(parsed.error)}`);
    return;
  }
  const capabilities = parsed.data;
  const reasoningEffortMap = Object.fromEntries(
    capabilities.reasoning.levels.map((level) => [level, level]),
  );

  pi.registerProvider("spark-cluster", {
    baseUrl,
    api: capabilities.api,
    apiKey: "none",
    models: [
      {
        id: capabilities.model,
        name: `${capabilities.model} (Spark cluster)`,
        reasoning: capabilities.reasoning.supported,
        thinking: capabilities.reasoning.supported
          ? { mode: "effort", efforts: capabilities.reasoning.levels }
          : { mode: "none" },
        input: capabilities.input,
        supportsTools: capabilities.tools.supported,
        contextWindow: capabilities.context_window,
        maxTokens: capabilities.max_output_tokens,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        compat: capabilities.reasoning.supported
          ? {
              thinkingFormat: capabilities.reasoning.request_format,
              supportsReasoningEffort: true,
              reasoningEffortMap,
              reasoningContentField: capabilities.reasoning.response_field,
            }
          : {},
      },
    ],
  });
}

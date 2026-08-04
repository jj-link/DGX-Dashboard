import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

const LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"] as const;

type ProviderOptions = {
  id: string;
  label: string;
  baseUrl: string;
};

type CapabilityLoad = {
  payload: unknown;
  error: string | null;
};

export async function loadCapabilityDocument(baseUrl: string): Promise<CapabilityLoad> {
  try {
    const response = await fetch(`${baseUrl}/model-capabilities`, {
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(5_000),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return { payload: await response.json(), error: null };
  } catch (error) {
    return { payload: null, error: error instanceof Error ? error.message : String(error) };
  }
}

export function registerCapabilityProvider(
  pi: ExtensionAPI,
  options: ProviderOptions,
  loaded: CapabilityLoad,
): void {
  if (loaded.payload === null) {
    pi.logger.warn(`${options.id} capability discovery unavailable: ${loaded.error}`);
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
      quantization: z
        .object({
          weights: z.string().min(1).max(32).regex(/^[A-Za-z0-9._+-]+$/),
        })
        .strict()
        .optional(),
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

  const parsed = capabilitiesSchema.safeParse(loaded.payload);
  if (!parsed.success) {
    pi.logger.error(`${options.id} capability discovery rejected: ${z.prettifyError(parsed.error)}`);
    return;
  }

  const capabilities = parsed.data;
  const reasoningEffortMap = Object.fromEntries(
    capabilities.reasoning.levels.map((level) => [level, level]),
  );
  pi.registerProvider(options.id, {
    baseUrl: options.baseUrl,
    api: capabilities.api,
    apiKey: "none",
    models: [
      {
        id: capabilities.model,
        name: `${capabilities.model} (${options.label})`,
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

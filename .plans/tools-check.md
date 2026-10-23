
Write and run a small Python script using this codebase’s existing Claude/OpenAI subscription integrations to test whether these two configurations are equivalent:

A: LLM(SYS_PROMPT1.with_tools([read, write]), SYS_PROMPT2)
B: LLM(SYS_PROMPT1, SYS_PROMPT2.with_tools([read, write]))

Adapt this pseudocode to our actual interfaces. Keep prompt text, prompt order, tools, model, and other settings identical.

Check two things:
1. Both configurations produce identical provider-facing model inputs, tool definitions, and caching configuration. Inspect the final serialized requests, ignoring only genuinely irrelevant transport metadata.
2. Switching between A and B preserves cache reuse, as demonstrated by the provider’s reported cached-input tokens—not response speed or matching generated text.

Use a sufficiently long, stable prompt to meet the model’s caching minimum. Warm the cache, establish an identical-request baseline, then test A→B and B→A. Account for provider cache granularity; a tiny shared-prefix hit alone does not prove success. Use bounded retries if necessary.

Accept a provider argument so Claude and OpenAI can be tested separately.

Print exactly `YES` to stdout only when both checks pass for the selected provider. Otherwise print exactly `NO`. Missing cache telemetry or inconclusive results must produce `NO`, with a brief explanation on stderr. `NO` means “not verified,” not necessarily “proven different.”

Reuse existing authentication and adapters. Never print credentials. Keep tool execution disabled or harmless, output tokens low, and the implementation simple.
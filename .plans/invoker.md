# invoker: providers are just handlers; ctx.get_invoker()

## Motivation
Main goal (user): "make it so invoking is simpler and clearer."
Today, picking a backend is split across three mechanisms: the `invoke_llm` override (OpenRouter), a per-context `invoke_llm=` field (OpenAI/Anthropic presets), and identity checks scattered around (`ctx.invoke_llm is not None`, `ctx.invoke_llm is not invoke_llm`, `NATIVE_WEBTOOLS[ctx.invoke_llm]`). User called the identity lookup "weak and weird".

User's words: "we should have a `handler` for invoke_llm. Then we just expose a function called `ctx.get_invoker() -> fn`, and we check it's being invoked by what we expect." "all openai models are invoked with openai provider automatically." "openrouter is a fallback handler with lower priority, openai and anthropic subs take priority. its just handlers."

Result: one handler chain decides the backend for every context, by model id. One function answers "who runs this ctx". No overrides, no presets carrying `invoke_llm=`.

## Design
Core (ex6.py):
- Replace the `invoke_llm` overridable with handleable `invoke_llm(ctx) -> fn | None`. Handlers return the generator fn that will run the ctx, or None to abstain. First truthy wins (normal `app.call`). Default: raise `RuntimeError(f"no provider handles model {ctx.model!r}")`. No dummy backend.
  - Handlers return a fn (not a generator) because a generator fn can't abstain.
- `Context.get_invoker()`: `self.invoke_llm or app.call("invoke_llm", self)`. Explicit `ctx.invoke_llm` field stays as an escape hatch (eg tests, forcing OpenRouter for an openai/ model).
- `Context.invoke`: `llm_fn = llm_fn or self.get_invoker()`.
- Traceability: `invoke` debug_prints `[invoke] {ctx.name} {ctx.model} -> {fn.__module__}.{fn.__name__}` per turn. Selection panel shows the invoker module next to the model.

Providers (each is just a handler):
- provider_openai.py: `@ex6.handler` returns its fn for `openai/` ids without a `:` suffix. Exports `WEB_TOOLS = [WEB_SEARCH]`.
- provider_anthropic.py: same for `anthropic/` ids without `:`. Exports `WEB_TOOLS = [WEB_SEARCH, WEB_FETCH]`.
- provider.py (OpenRouter): `@ex6.handler(order=100)`, claims everything (fallback). Its current `@ex6.override def invoke_llm` becomes a plain fn, eg `invoke_openrouter`. Exports `WEB_TOOLS = []`.
- Identity checks become `ctx.get_invoker() is <module fn>`: both provider footers, `supports_audio`.
- Disable subscriptions = don't load that plugin (or a project handler at lower order).

agents.py:
- Presets drop `invoke_llm=`; model id alone routes. Imports of provider invoke fns go away.
- `add_native_webtools(ctx)`: `inv = ctx.get_invoker()`; explicit `if inv is provider_openai.invoke_llm: ... elif inv is provider_anthropic.invoke_llm: ...` adds that module's `WEB_TOOLS`; else debug_print "no native web tools via <module>". Delete `NATIVE_WEBTOOLS`.
- Resolved at setup. If the model changes later, the provider raises "doesn't support native tools", loud not silent.

Direct callers of `app.get_implementation("invoke_llm")` switch to `ctx.get_invoker()`:
- commands.py `_llm_one_shot` (/cm)
- sigils.py

## Behaviour changes
- Every `openai/` and `anthropic/` model (explore_agent, /cm, project agents) now runs on subscriptions, not OpenRouter. EXPLORE_MODEL / SIGIL_MODEL are gemini: unaffected.
- `:online` and other suffixed ids stay on OpenRouter (websearch_agent).
- Without any provider plugin, invoking raises instead of streaming dummy "token" text.

## Cleanup (same change)
- Delete stale `.plans/websearch.md`, `.plans/native_tools.md`; update `_notes/websearch.md` to NativeTool design.
- AGENTS.md: `invoke_llm` moves from overrides to handlers (decorator list + overrides-vs-handlers example); `Context` line mentions `get_invoker()`; provider lines say "handler for openai/ ids" etc.; update add_native_webtools line.

## Files
- ex6.py: invoke_llm handleable, Context.get_invoker, Context.invoke, selection panel label
- _ex6/provider.py, provider_openai.py, provider_anthropic.py: handlers, WEB_TOOLS, footer/audio checks
- _ex6/agents.py: presets, imports, add_native_webtools
- _ex6/commands.py, _ex6/sigils.py: one-shot calls
- _ex6/_plugin_examples/*, inline.py docstring, agents_sound.py docstring: any `invoke_llm=` mentions
- AGENTS.md, _notes/websearch.md, .plans/*

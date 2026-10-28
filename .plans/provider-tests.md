# provider-tests: catch silent provider failures

## Motivation
User's words: "One thing that concerns me (deeply) is caching, and the hackiness of it all. If, one day, caching behaviour suddenly changes, i wont be aware of it, and my systems will fail silently, ill be paying extra token costs." "Likewise, if tool-inputs or outputs are malformed silently, i also won't know either."

The anthropic provider drives `claude -p` over undocumented stream-json + SDK-MCP. The openai provider impersonates codex_cli_rs against a private ChatGPT backend. Both can change under us with no error. The goal is a live smoke test that turns those silent changes into loud FAILs.

Scope (user): "It does NOT need to be a formal or complete test suite; less lines of code is better. It should just catch super basic stuff, and it should focus on anything that can fail silently."

Target: one file, `test_providers.py`, about 200 lines. Plain script, no pytest.

## Files
- `test_providers.py` (new, repo root)
- `ex6.py`: `LLMResult` (line ~630)
- `_ex6/provider_anthropic.py`: `_ClaudeProcess._result`, `pump`, `_processes`
- `_ex6/provider_openai.py`: `invoke_openai` (usage parsing, tool args `except: args = {}`)
- `_ex6/provider.py`: `invoke_openrouter`, `_log_invoke`

## Step 0: small core change (needed so tests can see caching)
Right now cached and cache-write tokens only go to logs and debug_print. `LLMResult` never carries them. Add:
- `LLMResult.cached_tokens: int = 0`, `cache_write_tokens: int = 0`
- `LLMResult.raw_usage: dict` holds the provider's native usage dict (debug only). This lets the test check the Anthropic `cache_creation.ephemeral_1h_input_tokens` without adding more fields.
- Fill these in from all 3 providers. The numbers are already computed there.

Optional, related: openai `except: args = {}` and openrouter `except: pass` swallow malformed tool-arg JSON. Make them loud (LLMResult.error or a debug_print with a canary string the test greps for).

## Harness
- `app = ex6.App()`. Don't call `_load_plugins`, because `agents.py` setup needs `app.tui`. Register only the provider modules (`provider`, `provider_openai`, `provider_anthropic`) via `app._register_module`. Set `ex6.get_folder()` aside if needed.
- Drive through the real path: `ctx.invoke(prompt)` with `yolo=True`, then poll `ctx.is_running()` with a timeout. The real path matters because the claude process's `seen` check depends on how ex6 appends messages.
- Record each turn's `LLMResult` by wrapping the invoker: `Context(..., invoke_llm=record(invoke_anthropic))`. The wrapper uses `yield from`, so `close()` still propagates.
- Each tool fn records the args it received in a list, so the test can compare them exactly.
- `check(name, ok, detail)` prints PASS/FAIL/WARN plus the numbers (cache ratio, overhead tokens). Numbers always print so drift stays visible. Exit code is non-zero on any FAIL.
- CLI: `python test_providers.py [anthropic] [openai] [openrouter]`. Default is anthropic + openai (subscription, $0). openrouter is opt-in because it costs real money.
- Cheap models: anthropic HAIKU_45, openai GPT_6_LUNA (or whatever is cheapest that works). Model per provider sits in a dict at the top.

## Checks (per provider)
1. basic: the system prompt says "reply only with <nonce>". Assert the reply contains the nonce (system prompt actually delivered), `error is None`, `input_tokens > 0`.
   - overhead: `input_tokens - estimate(system + user)` stays under a threshold (start around 1500, tune after the first run). Catches claude -p or codex silently injecting more context.
   - usage capture: `app.plugin_data["anthropic:usage"]["utilization"]` / `["openai:usage"]["percent"]` exists. Catches renamed headers/events that would make the footer show "(unknown usage)".
2. tool args round-trip: `echo_args(text: str, n: int, items: list[str])`. The prompt gives exact values: unicode, quotes, backslash, newline. Assert the recorded args equal the expected ones exactly (not `{}`, not a str).
3. tool results, big + parallel + empty: in one turn, ask the model to call `big_result()` and `noop()` in parallel. `big_result` returns about 80KB of filler with a nonce at the very end. `noop` returns "". Assert both were called in the same LLM turn, the final reply contains the nonce (no truncation or file-preview swap), and there was no error.
4. image in tool result: a tool returns a `ToolResult` with an `ImageAttachment` of a solid-color PNG (generated with zlib+struct, no PIL). Assert the reply names the color. Catches attachments being dropped silently.
5. caching (the main one):
   - padded deterministic system prompt (about 6k tokens, above every model's min cacheable size) + the tools above.
   - turn 1: user msg. turn 2: follow-up user msg. turn 3: a turn that calls a tool (tool loop).
   - anthropic: turns 2 and 3 have `cached_tokens / input_tokens >= 0.8`. Their cache_write stays small (no full prefix rewrite). Turn 1 has `raw_usage.cache_creation.ephemeral_1h_input_tokens > 0` (1h TTL env var still works).
   - anthropic: `_processes[ctx]` is the same object across all turns, and the debug buffer has no "history diverged". A silent process restart means a lost cache plus a flattened transcript.
   - openai: caching is best-effort. FAIL if turns 2 and 3 both have 0 cached tokens, WARN if the ratio is < 0.5.
   - openrouter (opt-in): same thresholds as anthropic, for an anthropic/ model with `cache_manually`.
6. canaries: at the end, grep `app.debug_buffer` for `invalid output`, `unexpected control request`, `stalled`, `exception`, `diverged`, and FAIL with the matching lines. These are the provider's own "something is off" signals, and right now nobody reads them.

## Not doing
- No offline unit tests of the converters (`msg_to_dict`, `_to_responses_input`). The risk is upstream drift, not our pure functions.
- No retries or flake handling beyond the openai WARN/FAIL split.
- No native web-tool test (it costs quota and is visible when it breaks). Add later if wanted.

## Open questions
- Run it automatically? For example a `/test_providers` command, or a cheap subset at the start of `run_horizon`. Until then it's manual.
- Append each run's numbers to `.ex6/provider_tests.jsonl` so drift is visible over time? It's about 3 lines.

## QUESTION ANSWERS:
- run it manually, have a `run_tests()` func. llms can run it themselves.
- yeah, do that. (thatll use ex6.get_folder() right? maybe makes sense to do `.ex6/provider_tests/test-logs.jsonl).  Also make sure you document it at the top of the file so agents become aware of it, and can read it.

# invokelogs: extract insights from logged invokes
Status: active

## Motivation (user's words)
"I wasn't aware that I was collecting so many invoke calls."
Use this data to try to extract insights. "Hopefully we can use these insights to make our agents more useful."

## Remaining work
Logging, dedupe storage and auto-culling are done. Data has piled up: `ex6.get_folder("provider_util") / "invokes.db"` is ~107 MB as of 2026-11-04.
Use it to extract insights that make our agents more useful. The insights are plain SQL queries over the DB, eg cost per ctx, context bloat per tool, error rate per model.

## DB layout (for writing queries)
```
blobs(hash TEXT PRIMARY KEY, data TEXT)          -- one row per unique message / tools list
invokes(id INTEGER PRIMARY KEY, ts, cwd, ctx, model, provider,
        input_tokens, output_tokens, cached_tokens, cache_write_tokens, cost,
        finish_reason, error,
        messages TEXT,   -- JSON list of blob hashes, in order
        tools TEXT,      -- blob hash of the tools list
        output TEXT)     -- JSON: response text, tool_calls, native_calls
```
- Messages are the provider-format payload, ie what was actually sent, not ex6 `Message`s.
- Anthropic logs `_ClaudeProcess.transcript`: the context window as ex6 sees it. claude-code adds its own bits on top.
- Dynamic content (eg the `<context_usage>` reminder) changes a message's hash, so near-duplicate blobs are expected.

## Files
- `_ex6/provider_util.py` (`log_invoke`, db, cull, /usage)

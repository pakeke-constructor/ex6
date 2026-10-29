# invokelogs

## Motivation (user's words)
"I wasn't aware that I was collecting so many invoke calls."
Use this data to try to extract insights. "Hopefully we can use these insights to make our agents more useful."
Storage must be "idempotent, robust, simple and painless."

## Current state
- `%APPDATA%\ex6\logs\invoke-*.txt`: ~62,600 files, ~4.2 GB (2026-02-01 to 2026-10-28). Listing the folder takes 30+ s.
- Written by `_log_invoke` in `_ex6/provider.py`, called from provider.py, provider_openai.py, provider_anthropic.py.
- Problem: "most of the stuff is badly formatted, it doesn't show output upon completion, it only shows the initial system-prompt a lot of the time."
- Size is mostly quadratic: each invoke in an agent loop resends the full history, and each log dumps all of it again.

## Tasks
1. Clear the folder and start fresh. The old logs can safely be deleted (`%APPDATA%\ex6\logs\invoke-*.txt`). This replaces the "cull empty files" idea.
2. FIX (must be fixed): several call sites pass `[]` as messages, so their logs have a header but no conversation:
   - `_ex6/provider_anthropic.py:397`
   - `_ex6/provider_anthropic.py:553`
   - `_ex6/provider_openai.py:167`
   Every call site must log the actual messages sent.
3. Going forwards, keep logging invocations, but log on completion, not on invoke (include the model's output). Replace `_log_invoke` with the SQLite store below.
4. Automatic culling of old logs (user: "Set like a storage limit (eg 1gb) and then cull by how old the stuff is automatically").
   - `_DB_LIMIT = 1_000_000_000` bytes in provider.py.
   - Live size = `(page_count - freelist_count) * page_size`. Freed pages get reused, so the file stays at ~limit with no VACUUM needed.
   - Checked after every logged invoke (cheap pragma). Over limit: delete oldest 10% of invokes, GC unreferenced blobs (`json_each(invokes.messages)` + `invokes.tools`), repeat until under 90% of limit. The 90% target stops it re-culling on every invoke.
5. Once fresh logs accumulate, use them to extract insights to make our agents more useful.

## Status (2026-10-29)
- Tasks 1-4 done. providers_test (anthropic, openai) passes and writes complete rows to invokes.db.
- Anthropic logs `_ClaudeProcess.transcript`: system + everything sent to the claude process + its replies. That's the context window as ex6 sees it (claude-code adds its own bits on top).
- Leftover: running ex6 instances on old code still write `logs/invoke-*.txt`. Restart them, then delete `%APPDATA%\ex6\logs`.
- Next: task 5 (insights), once data accumulates.

## Storage design (agreed)
User's idea: hash strings; store either directly or as a reference to a reused blob (eg sys-prompt).
Refinement: hash at message granularity. Every message is a reference, so there's no inline-vs-reference choice. Sys prompt, earlier turns, tool results and the tools list all dedupe automatically; each invoke only adds its new messages, so storage grows linearly.

- Single SQLite file: `ex6.get_folder() / "invokes.db"` (= `%APPDATA%\ex6\invokes.db`). Stdlib `sqlite3`, no deps.
- Schema:
  ```
  blobs(hash TEXT PRIMARY KEY, data TEXT)          -- one row per unique message / tools list
  invokes(id INTEGER PRIMARY KEY, ts, cwd, ctx, model, provider,
          input_tokens, output_tokens, cached_tokens, cache_write_tokens, cost,
          finish_reason, error,
          messages TEXT,   -- JSON list of blob hashes, in order
          tools TEXT,      -- blob hash of the tools list
          output TEXT)     -- JSON: response text, tool_calls, native_calls
  ```
- Hash: sha256 of `json.dumps(msg, sort_keys=True)`.
- Idempotent: `INSERT OR IGNORE` into blobs. Same content always gives the same row.
- Robust: WAL mode (several ex6 instances write at once). One transaction per invoke, so no half-written records.
- Simple: ~40 lines, replaces `_log_invoke`. Reuse `_log_safe` for anything that isn't JSON-safe.
- Log the provider-format payload (openai input_items / openrouter messages / anthropic), ie what was actually sent, not ex6 `Message`s. Goal is to see the real context window. (User confirmed.)
- Accepted (user: "keep it legitimate and correct"): dynamic content (eg `<context_usage>` reminder appended to tool results) changes the hash and gets stored again. Small cost, and it's what was actually sent.
- Insights (task 4) are just SQL queries over this DB (cost per ctx, context bloat per tool, error rate per model, etc).
- toolstats (`.plans/toolstats.md`) stays as JSONL for now. It could move into this DB later.

## Files
- `_ex6/provider_util.py` (`log_invoke`, `_log_safe`, db + cull); `_ex6/provider_openrouter.py`
- `_ex6/provider_openai.py`
- `_ex6/provider_anthropic.py`
- `ex6.get_folder()` in `ex6.py`

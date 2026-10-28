# toolstats

## Motivation
Count how often tools are called, saved across all sessions, for observability:
"is this tool used?", "make a pie chart of tool calls", "which tools take up the most space?",
"which models call the most tools?" etc.

## Design (agreed)
- New plugin `_ex6/tool_stats.py` (~30-50 lines). No core changes.
- App-wide `@ex6.after_tool_calls` hook (fn(ctx)), so it catches tools from every file (tools.py, web_tools.py, tasks.py, skills.py), not only tools.py.
  - Fired in `Context` run loop (ex6.py ~line 1261) after `call_tools` returns. Read the last assistant message's `tool_calls` plus the matching `role="tool"` messages after it (by `tool_call_id`).
  - Also log `msg.native_calls` (provider web search/fetch), with tool name `native:<name>`.
  - Caveat (accepted): calls cut off by stop_early never reach history, so they aren't logged.
- Storage: append-only JSONL, one line per tool call, written immediately (no flush timer, no sqlite).
  - Monthly files: `ex6.get_folder() / "toolstats" / "YYYY-MM.jsonl"` (= `%APPDATA%\ex6\toolstats\2026-10.jsonl`).
  - NOT in `logs/` (that folder gets wiped).
  - No auto-deletion. Old months deleted by hand.
- Store raw events, not counts. Aggregate at read time.

## Fields per line
- `ts`
- `cwd`: context cwd (`ctx.cwd or os.getcwd()`)
- `ctx`: `ctx.name`
- `model`: `ctx.model`
- `tool`: tool name
- `args`: tool args, but DON'T log multiline strings (eg edit-tool content). Drop them.
- `error`: True/False (result text starts with `ERROR:`)
- `preview`: first few characters of tool output (single line, ~80 chars)
- `bytes`: total bytes of tool output

## Command
- `/toolstats [month|all]`: counts per tool sorted by frequency (default: current month). Keep it simple; deeper questions (pie charts, per-model, bytes) answered by pointing an agent at the jsonl.

## Files
- `_ex6/tool_stats.py` (new)
- `ex6.py`: `get_folder()`, `Context` run loop (after_tool_calls), `call_tools`, `Message` (tool_calls, native_calls, tool_call_id)

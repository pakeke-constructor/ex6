# rewind: agent-controlled context rewind (checkpoint / rewind)
Status: active

## Motivation
Agents can't manage their own context except via subagents (briefing cost, no shared context) and handoff (nukes everything).
User idea: agent "undoes" wasted exploration:
```
checkpoint("Starting on fixing auth refresh...")
read a.py [4k] / read b.py [3k] / search [1k] / read c.py [3k]
rewind("api for doing xyz is in c.py, lines (114-150). It's a SSOT through this file, a.py and b.py call it.")
--> prunes messages back to the checkpoint.
```
Framing: checkpoint/rewind = an *inline subagent*. Full prior context, shared cache prefix, zero briefing; only the conclusion survives.

## Design
Tools (plugin, `_ex6/rewind.py`; no core change):
- `checkpoint(note)` -> returns "checkpoint #N: note". Stack; nested allowed. Index stored in `ctx.data_volatile["rewind:stack"]`.
- `rewind(summary, keep=[])` -> pops latest checkpoint. Applied in `after_tool_calls` hook (llm_suspended, loop continues):
  - delete messages after the checkpoint's tool-result message
  - rewrite checkpoint's tool result to: note + summary + auto stub + kept snippets
  - must be the only/last tool call in its turn (else error result, no rewind)
- No checkpoint on stack -> fallback to last user message (implicit checkpoint). Covers agents that forget to checkpoint.

Auto stub (deterministic, agent can't forget it):
- one line per rewindd call: `read_file a.py`, `search ".auth_token()"`, ... (reuse render_tool_call-ish names)
- MUTATIONS listed loudly: `edit_file a.py +3 -2`, shell calls with tool_summaries `!`. rewindd edits still happened on disk.
- tokens freed, so the model gets feedback that it worked.

`keep`: list of `path:start-end`; re-read fresh at rewind time and inlined. Avoids agent paying output tokens to copy code into summary ("read a short snippet of c.py").

## Risks to watch
- Negative knowledge loss -> re-exploration loops. Tool docstring must demand "what was NOT useful"; stub list helps.
- Models trained append-only: will under-use it. Nudge via `<context_usage>` reminder / ENV_PROMPT line, not more tools.
- anthropic provider: divergence -> process restart, resumed from a synthesized session file (`_write_session`, structured history kept). Costs a cache miss + process startup per rewind.
- Transparency: rewindd messages vanish from UI. Option: keep them in `data_volatile["rewind:archive"]` and render a collapsed row. (invokes.db already logs them if LOG_INVOKES.)

## Files
- `_ex6/tools.py` (handoff for pattern, render_tool_call rows)
- `_ex6/tool_summaries.py` (`!`/`?` mutation flag)
- `ex6.py` Context: edit_messages, after_tool_calls hook, invoke loop (~1361)
- `_ex6/provider_anthropic.py` `new_messages` / `_write_session`

## Later / side ideas
- `forget(call_ids)`: non-contiguous drop (keep c.py read, drop a/b) — needs model-visible call ids.
- auto-stale: older read of a file gets replaced with `[stale, re-read later]` when re-read/edited. Zero agent judgment.
- per-tool-result TTL: big results auto-collapse after K turns unless referenced.

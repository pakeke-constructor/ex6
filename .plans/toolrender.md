# toolrender

STATUS: API structure NOT decided yet. Candidates below; confirm with user before implementing.

## Motivation (user's words)
Current tool rendering: `[>] bash('sed -i \'s/return not p.is_alive(), 1.0..., 120)`
Fine, but missing crucial info:
- "How many tokens did this incur?"
- "what's actually happening in this command?"

Current render-tool-line API is "a bit weird/ugly... and hard to extend". Want a clean, extensible API.

## Desired behaviour
Token count replaces `[>]`: `[950]`, `[2.1k]`, padded to fixed width, so `99` -> `[99 ]`.
(`[2.1k]` is 4 chars: either width 4, or drop the decimal for thousands: `[2k ]`, `[12k]`. Undecided.)

Tools customize their own rendering, e.g.
- `edit_file: "file.txt", +304 -250` (+ green, - red). User said read_file; +/- fits edit/patch better, read_file maybe `"file.txt" 1-304`.
- bash: a cheap GPT-luna agent summarizes the command's purpose:
  `bash: [!] running tests` / `bash: [?] checking for plugin usage`
  `[!]` = destructive (edits files, runs other programs), `[?]` = read-only. Good enough > perfect.

Token count is PER-TOOL (not cumulative). Per-tool answers "what did this cost", big calls stand out; cumulative = ctx size, belongs in footer, and blows the fixed width.
Count color by size (theme keys): muted -> success (green) -> warning (yellow) -> error (red).
Thresholds NOT final. Don't guess: BENCHMARK when implementing, since logging only started recently and more data is wanted.
How to benchmark:
- Read `<get_folder("provider_util")>/invokes.db` read-only (`file:...?mode=ro`). Use distinct `blobs` so resent messages aren't counted twice.
- Map call id -> tool name from assistant `tool_calls` (anthropic/openrouter) and `function_call` items (openai).
- Results: `role=tool` blobs (`tool_use_id`) and `function_call_output` blobs (`call_id`).
- Per call tokens = estimate(args) + estimate(result). Report overall p25/50/75/90/95/99/max, plus per-tool p50/p90.
- Choose thresholds so muted covers roughly the majority, green up to ~p90, yellow up to ~p99, red ~top 1%. Show the user the numbers and the proposed thresholds before hardcoding.
- read_image comes out undercounted (log strips base64), so image tokens need a real estimate.
Starting guess (from an early ~5h sample): <500 muted, <1.5k green, <=4k yellow, >4k red.
Since color now means size, status shows differently: running = spinner in the brackets, error = `[ x ]` or the tool name in red (undecided).

## Invariants (whatever the API)
- Core always draws the `[tok]` prefix / spinner. Renderers only own the text after it.
- Renderers return spans `[(text, theme_color_key), ...]`, not draw into buf. Core handles truncation/width.
- Tokens = estimate(args + result text), cached per row, not recomputed each frame.

## Candidate APIs
1. Fn attribute: `bash.render = fn(ctx, call, result) -> spans`. Simplest, colocated. Can't restyle others' tools/native tools.
2. Handler chain `@ex6.handler def render_tool_call(ctx, call, result) -> spans | None`. Default = current `name(args)`. Reuses handler system; one plugin can cover bash+powershell+native. (Leaning toward this.)
3. Execution-time label: `ToolResult(text, label=spans)`, stored on tool message as display-only (never sent to LLM). Computed once, survives fork. Good for edit_file +/-; doesn't cover running state.
4. `Tool(fn, render=...)` wrapper like NativeTool. Explicit but heavy; probably overkill.

Suggested combo: 2 as API, plus 3's display-only label for execution-time facts; default handler prefers stored label, else `name(args)`.

## Bash summary notes
- Must be async + cached (render runs every frame). Return None -> show raw command until ready.
- Cache keyed by tool call id, display-only on the message (survives fork) rather than data_volatile.
- Kick off at call start, not render time (avoid summarizing whole history on reopen).
- Cheap model outputs `!|? summary`.

## Relevant code
- ex6.py `ToolCall` dataclass (~L655)
- ex6.py `render_tool_line` (~L863) - current drawing
- ex6.py `_default_tool_row` (~L2070) - builds ToolCall from call + result
- ex6.py `render_work_mode` (~L2087) - where rows are built/drawn (incl. native_calls, pending tools)
- ex6.py `get_token_estimate`, `tool_result_text`
- _ex6/tools.py - edit_file, read_file, bash, powershell
- bash summarizer probably its own plugin, e.g. `_ex6/tool_summaries.py`

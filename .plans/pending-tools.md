# Pending tools

## Motivation and approved scope

User: minimal, clean code is best; less is better. Formalize in-progress tool
state while REMOVING rendering/history/execution bookkeeping. Committed messages
represent conversation history; incomplete tool work must not require inserting
messages and repairing them later.

User approved this shape:
- One PendingTool(call, result=None) record per requested call.
- One ctx.pending_tools ephemeral list; NO ctx.pending_message.
- call_tools owns this list and sets ctx.pending_tools = None in finally, after
  all started workers join. Never leave stale pending state, even on failure.
- Existing llm_current_output and llm_result already contain assistant turn.
- Threads remain local to call_tools, not on Context.
- Workers publish normalized results into their own records.
- Commit assistant plus tool results together after joins; skip on stop/truncate.
- Keep busy/stop/draining/deferred-clear behavior unchanged.
- No run-context, nonce, new lifecycle lock, state machine, compatibility layer,
  process helper, or unrelated refactor.

## Starting point / preserve edits

Implementation has NOT started. An attempted ex6.py patch failed before applying
anything. Only this plan was written by agent.
Latest working tree: ex6.py has user-added _poll_operation docstring (3 lines);
preserve it. .plans/pending-tools.md is untracked. Read fresh tree before editing.
Previous operations work is committed. User deleted test_handoff.py and
 test_operations.py; do NOT restore them. tests/test_patch_file.py now uses simple
assertions and run() at import, not unittest cases. Preserve that style.

## Relevant code

Everything production for this refactor is in ex6.py:
- ToolResult, Operation, Message, LLMResult, ToolCall
- _poll_operation, call_tools (overridable)
- Context fields, _read_llm_stream, _discard_stopped_tool_batch, invoke,
  truncate, clear, _clone
- _default_tool_row, render_work_mode

Current behavior:
- invoke.do_llm appends assistant before tools, then runs after_llm_turn hook.
- call_tools maintains local threads/results/started_ids plus ctx._active_tools.
  Tool results are appended only after every worker finishes.
- Renderer looks up historical tool messages, _tool_rows overrides and live
  threads. A completed sibling still appears running until all finish.
- invoke cleanup repairs stopped history via _discard_stopped_tool_batch.
- _tool_rows has no writers in current code/plugins; delete it and cleanup.
- _tools_invalidated is set by truncate/clear; retain it for this refactor.
- after_tool_calls runs after call_tools returns successfully; retain timing.

## Implementation

1. Add tiny dataclass PendingTool:
   call: dict
   result: ToolResult | None = None
   None means incomplete; ToolResult("") means completed with empty output.
   No thread, lock, error flag, done flag, status cache, Operation field.

2. Replace Context._active_tools / _tool_rows with:
   pending_tools: list[PendingTool] | None
   default None, init=False, repr=False (ephemeral, not serialized).
   _clone resets it to None. clear must not erase live records while draining;
   call_tools remains responsible for their lifetime.

3. Add one small Context._assistant_message(tool_calls) helper:
   builds assistant Message from existing llm_current_output (text content,
   copied chunks, supplied calls). No stored pending assistant.
   Share it for ordinary completion, batch commit, and pending rendering.

4. invoke.do_llm:
   read stream, stop check; append assistant immediately ONLY if no tool calls.
   Tool-call assistant stays in existing llm_current_output/llm_result until
   batch finishes. Keep after_llm_turn hook here.
   Delete _discard_stopped_tool_batch and its cleanup call entirely.

   Explicit approved hook change: on tool-call turns, after_llm_turn does not
   see assistant in get_messages() yet. It inspects llm_result and
   llm_current_output. Do not fake a history message for compatibility.

5. call_tools:
   - No calls => False as before.
   - Build local pending list for calls, publish as ctx.pending_tools.
     Keep this local list as authoritative for workers/commit.
   - Start ordinary threads. Preserve arg validation/tool_call_id injection,
     unknown-tool error, Operation polling/cancel behavior, debug logging.
   - Each worker converts result to ToolResult (preserve attachments) and
     enforces MAX_TOOL_OUTPUT_CHARACTERS BEFORE publishing result. Errors become
     ERROR: text as before. Unknown calls publish completed error too.
   - Remove results dictionaries, unused error fields, started_ids and
     _active_tools registration/cleanup.
   - Join every started worker even if launch/coordinator raises. Small
     try/finally, not abandonment or early cleanup while workers remain alive.
   - Under existing _msg_lock, after joins: stop => False/no commit;
     _tools_invalidated => reset flag and True/no commit; otherwise extend
     _messages with assistant + ordered tool Messages as one complete batch.
   - Clear ctx.pending_tools in finally on every exit, after joining. Ensure
     commit and clearing pending are observed consistently by renderer using
     existing message lock. No duplicate live and committed batch in one frame.
   - Rendering during draining may keep live records; they disappear after join.

6. Rendering:
   - Snapshot history and pending state consistently under existing _msg_lock.
   - Render committed history normally; append derived pending assistant and
     its records without adding them to _messages.
   - Reuse same message/row path where straightforward. Pending result lookup
     uses records directly (no transient tool Messages required just to render).
   - _default_tool_row should take call + result, not ctx + tool Message/thread.
     None => running; completed result => error if ERROR: prefix else ok.
     Preserve existing args/detail formatting, including empty output.
   - Delete _tool_rows override branch. Keep ToolCall display type/output hooks.
   - Existing streaming display remains; avoid duplicate assistant text/cursor
     while pending batch is displayed. Do not broaden streaming/COT behavior.
   - Historical tool-result lookup remains necessary; no new registry.

7. Cleanup:
   Delete _drop_tool_rows and calls from truncate/clear; simplify these methods
   accordingly. Truncate's message deletion and _tools_invalidated update should
   use existing lock so batch commit sees invalidation. Do not add lifecycle
   machinery. Reset pending_tools on clone, not share live records.

## Verification (small, direct tests)

Add tests/test_pending_tools.py in current simple assert/run style. Avoid large
fixtures, long mock hierarchies, or resurrecting removed tests. Events and captured
worker/run threads are sufficient to deterministically join/release test work.
Always release blocked work on assertion failure.

Cover essentials:
- Two siblings: completed empty/attachment/error result visible while slow one
  is still None; history contains no incomplete assistant/tool batch.
- Render pending calls with actual completed/running statuses, then completed
  history without duplicate rows. Use simple ScreenBuffer or capture tool rows.
- Successful join commits assistant + ordered tool results; pending_tools None.
  Include unknown call and output normalization/limit if cheap.
- Stop/clear/truncate: workers genuinely drain; no partial batch committed;
  pending_tools None after completion. Stop keeps busy until joins. Reuse one
  scenario/parameterization, not duplicate scheduling/handoff infrastructure.
- after_llm_turn sees llm_result/output but no incomplete history batch;
  after_tool_calls sees committed batch.
- Clone during pending work has pending_tools None.
- Verify cleanup on ordinary tool error; tools still return ERROR: result.

Run python -m unittest discover -s tests (imports execute assert-style tests),
and optionally python -m pytest if available. Inspect git diff --check, diff,
status and net production line counts. Refactor should remove code overall; if
it starts growing substantially, step back and simplify rather than introducing
abstractions. Do not claim behavior that wasn't exercised.

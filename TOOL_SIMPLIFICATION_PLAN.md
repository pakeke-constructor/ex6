# TOOL_SIMPLIFICATION_PLAN

## Goals
- Greatly simplify tool execution + rendering flow.
- Introduce formal `ToolCall` object for runtime/render state.
- Remove code_mode special rendering path.
- Keep one rendering mechanism in core `ex6` override path.
- LESS CODE = BETTER.
- Readability: No more crazy tabl[i]["id"].call_args stuff. It should be simple and readable.

## Non-goals
- No new features.
- No tool protocol redesign with providers.
- No broad refactors outside tool pipeline.

## Current complexity (why hard now)
- Tool state split across:
  - `llm_result.tool_calls` (raw dicts)
  - `ctx._active_tools` (threads)
  - `ctx._tool_renderers` (per-call custom render fn)
  - `ctx.messages` role=`tool` (final output)
  - `ctx._tools_invalidated` (truncate/clear coordination)
- Rendering path has mixed line types (text + callable renderers).
- code_mode has privileged custom UI via `ctx.set_tool_renderer(...)`.

## Target model (simple)
Single core concept for runtime/render rows:
- `ToolCall` dataclass with:
  - `id: str`
  - `name: str`
  - `args: list`
  - `kwargs: dict`
  - `status: Literal["running","ok","error"]`
  - `detail: Optional[str]`
- Context stores tool rows only as data:
  - `ctx._tool_rows: dict[str, list[ToolCall]]` keyed by assistant tool_call_id.
- Core renderer draws `ToolCall` rows using one function (`render_tool_line`).
- Plugins (including code_mode) may populate `ctx._tool_rows[...]` with `ToolCall` objects, but never custom render fns.

## Plan

### Phase 1: Add formal ToolCall type
1. Add `ToolCall` dataclass in `ex6.py` near `LLMResult`.
2. Keep existing behavior; only add type and helper comments.
3. Do not remove old paths yet.

Acceptance:
- code compiles.
- no behavior change.

### Phase 2: Replace renderer callbacks with row data
1. Add `Context._tool_rows`.
2. Remove `Context.set_tool_renderer`.
3. Update render path in `render_work_mode`:
   - for each assistant message with `tool_calls`, gather rows from `ctx._tool_rows[id]`.
   - if absent, build default single-row `ToolCall` from core state/message.
   - render rows via `render_tool_line` only.
4. Remove callable-based tool row insertion for tools.

Acceptance:
- tool rows still visible (running/ok/error).
- no `_tool_renderers` usage remains.

### Phase 3: Unify cleanup lifecycle
1. Remove `_tool_renderers` field entirely.
2. Update `truncate`, `clear`, `fork` cleanup/reset logic to use `_tool_rows` only.
3. Ensure removed messages also remove matching `_tool_rows` entries.

Acceptance:
- truncate/clear/fork no stale tool row state.

### Phase 4: Remove code_mode privilege
1. In `_ex6/code_mode.py`, delete `ctx.set_tool_renderer(...)` usage.
2. If code_mode wants multi-row display, write `ToolCall` objects into `ctx._tool_rows[tool_call_id]`.
3. Keep rendering owned by core only.

Acceptance:
- code_mode displays through same core renderer as all tools.
- no code_mode custom draw fn remains.

### Phase 5: Trim remaining coordination complexity
1. Reassess `_tools_invalidated` necessity after callback removal.
2. If safe, remove `_tools_invalidated` and related branching from `call_tools`.
3. Keep minimal synchronization: threads join -> append tool messages.

Acceptance:
- `call_tools` shorter and easier to follow.
- behavior same for stop_early/truncate/clear.

## Implementation constraints
- Prefer small commits per phase.
- After each phase:
  - run `python -m py_compile ex6.py _ex6/code_mode.py`
  - run quick manual smoke: one normal tool call + one code_mode run_tools call.
- No extra abstractions unless duplication painful.



## OK: I'm the Human, and im here to discuss the important stuff:
I have removed code_mode entirely. This should allow you to tear out a bunch of the complexity, and make stuff really, REALLY simple.

Please take a step back and try to make it really simple.

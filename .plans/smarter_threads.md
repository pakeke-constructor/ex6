# Tool-thread lifecycle and safe scheduling

## Motivation

User: `_active_tools` is ugly; there is no real difference between a stale tool and a dead tool. Make ownership and lifetime explicit instead of adding handoff-specific patches.

ex6 should stay a thin, plugin-controlled harness. Core owns execution and safe state transitions; plugins own what transitions do. A stopped run must not leave invisible workers that can mutate a new conversation. `schedule()` must mean a genuinely safe boundary, not merely `llm_is_running == False`.

## Current state / recovery notes

Work is uncommitted. Keep existing changes unless deliberately replacing them:
- `_ex6/tools.py:handoff` schedules a callback that calls `ctx.clear()` then `ctx.invoke(txt)`.
- `_ex6/agents.py` exposes handoff in MAIN_TOOLS.
- `Context.schedule(fn, *args, **kwargs)` allows one pending callback, rejects a second, executes immediately when idle, and otherwise exits the loop and calls it after cleanup. `clear()` cancels the pending callback.
- `tests/test_handoff.py` covers scheduling and handoff; last test run passed 9 tests.
- User edited schedule's docstring after implementation. Read latest code; preserve their wording unless changing semantics requires an update. They want short, plain-English docs.

Problems:
- `call_tools()` returns early on stop and its `finally` removes every batch entry from `_active_tools`, including still-live threads.
- Current stop check is `if ctx.stop_early and ctx._scheduled is None: return False`. This makes safety depend on whether a callback happens to be pending. Scheduling after early exit can run while abandoned tools are still alive.
- `_default_tool_row()` assumes a missing result means running, so dead/abandoned calls can render as running forever.
- `clear()` sets model idle without waiting for the model worker. `invoke()` resets shared stop state; old workers can resume or overwrite new-run state.
- A raw tool-call ID is not enough ownership if invocations reuse IDs.
- `_clone()` shallow-copies Context before resetting some runtime fields; scheduled callbacks and other new execution state must not leak into forks.
- Tool functions receive the real mutable Context. Discarding stale return values alone cannot prevent late writes, UI changes, or filesystem side effects.

## Direction: smallest safe baseline

Separate two facts rather than inventing one status that conflates them:
- Thread lifetime: still executing vs finished.
- Ownership: belongs to current run vs stopped/superseded run (stale).

A stale tool can still be alive. A cancellation request is not proof that it stopped. Python threads cannot safely be force-killed.

Recommended baseline: do not start a new invocation or run a scheduled mutation while previous context workers are still alive. Stop requests cancellation immediately; execution remains busy/draining until workers finish. This avoids pretending arbitrary tools can be isolated from Context without changing the plugin API.

If prompt reuse while non-cooperative tools continue running is required, ask before implementing. That requires a different design (isolated tool context / explicit write permissions), not just run IDs.

## Implementation steps

1. Establish execution ownership.
   - Use one small per-invocation record for identity, model thread, cancellation signal, and tool executions. Prefer this over accumulating unrelated Context flags.
   - Give each tool execution its owner, thread, completion/outcome, and cancellation/stale information. Keep execution bookkeeping separate from display `ToolCall`.
   - Reuse existing structures where possible. No executor framework, forced thread killing, or long-lived history registry.

2. Fix tracking before changing scheduling.
   - Register a tool before starting it; mark completion in its worker's `finally`.
   - Never remove a live thread merely because the caller stopped waiting.
   - Keep ownership until the run is fully drained. Preserve outcome long enough to render finished/error/stopped calls correctly, then clean it up with history/run cleanup.
   - Use run ownership as well as tool ID so old completion cannot delete a newer entry.
   - Commit tool results only for the owning, valid batch. A stopped/discarded batch must not append late results.

3. Unify stop and draining.
   - Add one stop-request API and route `/stop`, Ctrl-X, and clear through it instead of writing/resetting shared stop flags independently.
   - Cancellation belongs to the run and stays set; a new invocation cannot reset an old worker's cancellation state.
   - Request stop without blocking the UI. Reject invocation while old workers are draining; callers wanting a transition use schedule.
   - Keep running/busy status true until actual execution ends. Old cleanup must never reset a newer run's state.
   - Do not hold `_msg_lock` (or an execution lock) while joining threads, calling providers/tools/hooks, or running scheduled callbacks.

4. Make schedule depend on actual execution, not a boolean.
   - Keep one pending function and the existing second-call error.
   - Run immediately only if there are no model/tool workers left.
   - Otherwise finish/drain the owning run, complete all old-run cleanup, remove the pending function, then invoke it outside locks. Old loop does nothing afterwards.
   - Scheduling alone does not cancel tools. Remove the special `_scheduled` condition from the tool wait loop; one lifecycle rule should apply regardless of scheduling timing.
   - Define clear during a run as a stop request with clearing deferred until safe; decide how this interacts with an already-pending callback before coding. Preserve existing clear-cancels-schedule semantics unless explicitly changing it with user approval.
   - Keep handoff implemented in plugin via schedule; no `_handoff_prompt` or handoff logic in core.

5. Align UI and plugin waits.
   - Render executing, finished, failed, and stopped/stale calls from execution state, not absence of a result. Avoid labelling an alive canceled tool as finished.
   - Question/approval tools should exit when their owning run is canceled and remove only their own UI entry.
   - Blocking subprocess/network tools can finish naturally for this MVP. Document that a hung tool delays drain; do not silently detach it and claim the context is safe.
   - A tool must not wait for a scheduled callback: callback waits for that tool. Document this briefly in schedule if needed.

6. Reset lifecycle state on clear/fork.
   - Fork copies conversation/configuration, not threads, pending callbacks, cancellation state, or in-flight execution records.
   - Clear removes conversation/tool display state only when old workers cannot repopulate it.

## Tests

Use Events/barriers rather than relying on timing:
- Stop during parallel tools: live tools stay tracked until actual completion.
- A finished tool and a stale-but-live tool are distinguishable in bookkeeping and UI.
- Schedule before stop, during drain, and after model completion: callback never runs with a live old worker.
- Handoff waits for sibling tools, clears their late state/results, then starts one clean invocation.
- Scheduling twice rejects the second without losing the first; callback starts a new run without old cleanup clobbering it.
- Invoke while draining cannot reset cancellation or overlap workers.
- clear while executing and stop/clear races with result commit and scheduled execution.
- Reused tool IDs cannot let an old completion remove current tracking.
- Canceled UI tools unblock; forks do not inherit pending functions/runtime state.
- Check full diff and run all tests.

## Relevant files / scope

- `ex6.py`: Context runtime fields, invoke/schedule/clear/_clone, call_tools, _default_tool_row, Ctrl-X.
- `_ex6/commands.py`: stop/clr and any run-state checks.
- `_ex6/tools.py`: handoff, ask_user, ask_user_question, approve; subprocess/subagent waits only if required.
- `tests/test_handoff.py`: existing uncommitted tests; extend/replace for lifecycle contract.
- `.plans/stop.md`: related earlier plan for immediate provider cancellation and valid-history rollback. Do not implement its async-provider overhaul as part of this task.

Before editing, settle clear/pending-callback semantics and verify this design remains small. If it needs more than 300 new lines, stop and ask how to simplify. Focus on correct ownership, honest tracking, and safe draining; not immediate remote cancellation, undoing tool side effects, or a general task scheduler.

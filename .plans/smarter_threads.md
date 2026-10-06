# Owned tool execution, cancellation, and safe transitions

## Motivation

User: `_active_tools` is ugly; make ownership and lifetime explicit instead of adding handoff-specific patches. Less code changed is better.

ex6 stays a thin, plugin-controlled harness. Core owns execution and safe transitions; plugins decide how work runs and stops. Python threads cannot safely be force-canceled. A cancellation request does not mean a worker died, and dropping tracking cannot make Context safe to reuse.

Tools receive the real mutable Context. Discarding late results alone cannot prevent late conversation/UI writes or filesystem effects. Keep the context busy until its old work actually finishes; never silently abandon workers and admit a new invocation.

Implement this together with `.plans/operations.md`. That file specifies the Operation API and adapters; this file specifies lifecycle and transition safety. The agreed API is `Operation(cancel, poll)`, not start/wait callbacks, Context registration, or process isolation.

## Operation contract

```python
@dataclass
class Operation:
    cancel: Callable[[], None]
    poll: Callable[[float], str | ToolResult | None]
```

Tool creates its resource and returns closures capturing it. No start callback or handle plumbing. Keep ToolResult as completed text/attachments, separate from Operation.

- Plugin implements `poll(wait=5)` in seconds: waits up to deadline and returns None while pending, or str/ToolResult when finished.
- `poll(0)` checks immediately, never blocks. Empty string is a completed result; test `is None`, not truthiness.
- Core polls on the tool worker with short bounded intervals, never UI thread; no concurrent polls or new thread per poll.
- `cancel()` requests termination, must be quick, and may run concurrently with poll. Core calls it at most once.
- Continue polling after cancellation until underlying work and cleanup actually finish. Then join worker before cleanup/transition.
- Cancellation during startup stays recorded. Cancel Operation once it becomes available, if still unfinished.
- Startup is ordinary uncancellable Python; keep it short. Plugin owns resource cleanup before return, poll deadline compliance, and effective cancellation.
- Plain string/ToolResult tools remain supported and drain normally. No promise of forcibly terminating arbitrary Python, undoing side effects, or promptly interrupting a blocked provider.

## Current code / recovery

These are inspection notes, not implementation already performed. Verify latest code before editing.

- `call_tools` uses `_active_tools` keyed by raw tool ID; early stop can remove still-live threads. Its `_scheduled` exception makes safety depend on whether a callback is pending.
- `invoke` transforms/appends prompt before admission checks and resets shared stop state.
- `clear` resets immediately while workers may still be alive.
- `schedule` supports one pending callback. Idle callback runs on caller; deferred callback runs on model-loop thread after cleanup.
- `handoff` is already a plugin using schedule to clear/reinvoke. Keep it there; no handoff state in core.
- `_clone` shallow-copies runtime fields; new lifecycle state must not leak into forks.
- UI answer handlers use `ui_stack.pop()`; several waits do not observe stop.
- `bash`/`powershell` still use subprocess.run; Operation migration is specified in operations plan.
- `tests/test_handoff.py` currently has five tests, including assertions on old tracking/shared flags that need updating.

Preserve unrelated user edits in `_ex6/agents.py`, `_ex6/tools.py`, and other files. Inspect working tree first. Preserve schedule's short plain-English documentation where semantics still apply.

## Implementation

### 1. Ownership and tracking

Use one small invocation record for identity, model thread, cancellation signal, and current tool executions. Each execution owns thread, published Operation, cancel-claimed state, completion, and result/error. Reuse existing batch structures; avoid parallel registries and duplicated status booleans.

Capture run in worker closures. Cancellation stays set on that record and cannot be reset by another invocation. Record identity establishes ownership; raw tool IDs are provider/display metadata, not lifecycle identity.

Register before starting workers. Mark completion in finally; retain live records until actual exit/join, even when conversation rows are truncated. Keep completed outcomes only with the associated batch/message, not an unbounded second history.

### 2. Stop and drain

Add one stop-request API. Route `/stop`, Ctrl-X, and clear through it; migrate direct writable shared stop flags and relevant plugin checks.

Stop requests cancellation without blocking UI. Coordinator scans all batch executions while using bounded thread joins, claims published Operations' cancellation under a short lock, and calls cancel outside locks. Do not let a slow first sibling prevent cancel dispatch to the others.

Workers resolve Operation via repeated bounded polls. Keep tracking/busy until every started worker finishes and is joined. Cancel failure is logged through debug_print, not retried, and must not end draining. Poll failure is terminal tool error only after plugin resource cleanup; core cannot infer resource lifetime from an exception.

Check cancellation before starting tools, committing provider output/results, running hooks, and starting another model turn. Stop during streaming must not launch tools from an already assigned LLMResult. Close provider iteration on early exit when supported; do not implement `.plans/stop.md` async-provider overhaul.

Never hold lifecycle/message locks while joining, polling, canceling, calling providers/tools/hooks, or running callbacks.

### 3. Admission and safe scheduling

Reject invoke while run is executing/draining or another thread owns a reserved transition. Reject before transforming prompt or appending user message. Reserve admission before launching model thread.

Preserve schedule behavior:

- One pending callback; second raises without replacing first.
- Schedule requests exit after current turn/batch, without canceling siblings or starting another turn.
- Stop does not cancel pending callback.
- Idle callback runs immediately on caller; deferred callback runs on model-loop thread.
- Safe boundary: tools joined, provider closed, hooks and old-run cleanup finished. Model-loop thread may only dispatch callback and return; no old-run writes afterwards.
- Tool cannot wait for scheduled callback: callback waits for tool.

Serialize admission, stop, schedule, clear, and callback claim with a short lifecycle lock. Reserve transition ownership through dispatch so external invoke cannot slip between cleanup and callback clear/reinvoke. Callback runs outside locks, may synchronously clear/invoke or schedule a nested callback. Release only its reservation, even on exception; never clobber a new run it starts.

### 4. Clear policy: confirm before coding

User has not approved the proposed semantics; discussion was redirected to cancellation API. Confirm this rule rather than silently treating it as decided:

- Clear cancels pending callback, requests stop, and defers reset until draining finishes.
- Repeated clears coalesce; schedule rejects while clear pending.
- Idle clear stays synchronous.
- Clear may cancel callback before claim. Once claimed it is executing; simplest policy rejects external clear during dispatch but permits callback's own clear.

If user changes rule, update both plans. Do not introduce general callback queue. Clear must not admit new work while old workers can repopulate state.

### 5. History, rendering, and UI

Commit completed tool batch atomically under message lock, checking ownership/cancellation/batch validity with consistent lifecycle lock ordering. Never append one result at a time across a stop/clear race.

Discard stopped batch's assistant tool-call message and rows as well as results; preserve earlier valid batches and latest user prompt. No dangling calls in next provider request. Truncate invalidates removed batch specifically; siblings cannot resurrect it. Execution tracking survives display removal.

Render completion/error from execution outcome, not missing result. Alive canceled tool is executing/draining, not finished. Tools skipped due to stop must not appear forever running.

Question, optional-answer, approval, and escalation waits capture owning cancellation signal and remove only their own draw function on answer/cancellation. Replace pop() in these paths. Canceled approval returns denial, never None. Keep synchronous approval helper for callers that use it before returning Operation; no wholesale UI operation rewrite.

### 6. Clear/fork reset

Clear conversation/tool rows and volatile state only after safe boundary. Fork copies conversation/configuration, not thread records, locks, cancellation, pending callback/clear, transition reservation, current output, or suspended state.

Existing explore_agent parent tool remains tracked until child wait/removal finishes. Child stop propagation is optional follow-up, not a reason to declare parent drained early.

## Relevant files

- `ex6.py`: Operation/ToolResult, call_tools, Context lifecycle/admission/schedule/clear/truncate/_clone, tool rendering, Ctrl-X.
- `_ex6/commands.py`: stop/clr and running checks.
- `_ex6/tools.py`: blocking UI helpers, shell Operation adapters, existing handoff/subagent behavior.
- `tests/test_handoff.py`: update old lifecycle assumptions and extend tests.
- `.plans/operations.md`: detailed API, plugin obligations, shell adapter requirements, and polling tests.
- `.plans/stop.md`: related provider cancellation work, explicitly out of scope.

## Tests and verification

Use Events/barriers with bounded waits and finally releases, not timing-only proofs:

- Plain tools unchanged; Operation polls None until completion; empty result completes; poll(0) immediate; positive deadline honored.
- Stop during startup remembered; during polling cancel called once; cancel request/failure does not release context before actual completion.
- Parallel live workers stay tracked regardless of schedule timing; every sibling gets cancellation.
- Finished worker and canceled-but-live worker distinct in bookkeeping/rendering.
- Invoke while draining rejected before transformation/append and cannot reset old cancellation.
- Schedule before stop/during drain/when idle obeys same boundary; schedule alone never cancels.
- Double schedule preserves first; nested schedule/callback exception/new invocation release only own transition state.
- Pause before callback claim: external invoke rejected; clear can cancel unclaimed callback. Claimed callback obeys agreed clear policy.
- Handoff waits for sibling late writes, clears them, starts exactly one clean run without old cleanup overwriting it.
- Stop during stream never launches old tool calls; discarded batches leave valid history.
- Stop/clear/result commit and truncate races cannot append half-batches or resurrect removed calls.
- Reused tool IDs cannot confuse run ownership or delete live tracking.
- Parallel UI dialogs answer/cancel only their own entry; canceled approval denies.
- Fork inherits no lifecycle state.
- Shell migration, if included: output/exit/timeout unchanged; repeated polls preserve output; termination reaps process tree before transition.

Implement ownership/drain and Operation handling first, then admission/clear, then history/UI adjustments. Reassess 300-added-production-line budget before expanding. Prefer existing loop over scheduler, background reaper, executor, arbitrary tool isolation, or forced thread killing.

Run entire test suite and inspect full diff/status. Use ex6.debug_print for diagnostics. No implementation requested until user asks to proceed.

# Explicit cancellable tool operations

## Motivation

ex6 is a thin, plugin-controlled harness. Core should own execution lifetime and safe transitions; plugins should control how their work runs and stops.

Python threads cannot safely be force-canceled. Dropping tracking or requesting cancellation does not mean a tool stopped. Tools need an explicit way to describe cancellable work without Context callback registration, thread-local lookup, mandatory subprocess isolation, or shell-specific core APIs.

User chose a small, general return value:

```python
def my_tool(ctx: ex6.Context):
    proc = subprocess.Popen(...)
    return ex6.Operation(
        cancel=proc.kill,
        poll=poll,
    )
```

Here `poll(wait=5)` waits up to five seconds for completion; `poll(0)` checks immediately. It returns None while pending, or str/ToolResult when finished. The subprocess adapter defines poll; Popen.poll itself only returns an exit code and does not implement this contract.

No `start` function or handle plumbing. Tool creates its resource; returned closures capture it. Core remembers stop during startup and cancels as soon as the Operation becomes available.

This extends `.plans/smarter_threads.md`, not a replacement for its safety requirements. Implement the smallest lifecycle changes needed to make Operation cancellation and scheduling honest. Less code changed is better. If production changes need more than 300 added lines, stop and discuss simplification.

## Scope and current state

No implementation has been made for this API. Existing code was inspected:

- `ex6.py:ToolResult` contains completed text and attachments. Keep it unchanged.
- `ex6.py:call_tools` launches one Python thread per tool. `_active_tools` maps raw tool IDs to threads. Stop can return early and erase tracking for live workers.
- `Context.invoke` currently admits overlapping invocations and resets shared `stop_early`.
- `Context.schedule` has one pending callback. Idle callbacks execute on caller; deferred callbacks execute on model-loop thread after cleanup.
- `Context.clear` currently resets immediately even if workers remain alive.
- `_clone` shallow-copies runtime state and resets only some fields.
- `/stop` and Ctrl-X write `stop_early` directly.
- Question/approval tools block on `ui_stack`; several answer handlers use `pop()`, which can remove another tool's dialog.
- `bash` and `powershell` use blocking `subprocess.run`.
- `tests/test_handoff.py` has five existing tests. Some assert `_active_tools`, direct `stop_early` mutation, or manually construct an allegedly running context; update these to real execution boundaries.

Preserve existing unrelated working-tree edits in `_ex6/agents.py`, `_ex6/tools.py`, and `.plans/smarter_threads.md`. Inspect fresh git diff before starting.

## Public API

Add a dataclass next to ToolResult:

```python
@dataclass
class Operation:
    cancel: Callable[[], None]
    poll: Callable[[float], str | ToolResult | None]
```

Purpose docstring, short and plain English:

> Cancellable work returned by a tool. Python threads cannot safely be killed; cancel requests termination of the underlying work. poll(wait) waits up to that many seconds and returns None while pending, or the completed result. poll(0) never blocks.

Contract:

- Existing strings and ToolResult returns continue working unchanged.
- Plugin supplies `poll(wait=5)`, with wait measured in seconds. Core repeatedly calls it with a short bounded interval on the tool worker, never on UI thread. No concurrent poll calls for one Operation.
- None means pending. str or ToolResult means finished, including an empty string or empty ToolResult. Never use truthiness to detect completion. Wrap a completed no-output result in ToolResult("") rather than returning None.
- `poll(0)` checks immediately without blocking. Positive wait blocks only until completion or the supplied deadline; returning None must not cancel or restart the work.
- A finished result means underlying work and cleanup are complete, not merely termination requested. Core stops polling after receiving it.
- `cancel()` is called at most once, potentially concurrently with `poll()`. It must be quick and safe if work has just finished. It requests termination; it does not prove completion.
- On cancellation before Operation publication, core calls cancel once it receives the Operation, then continues polling to drain it.
- Normal completion does not call cancel.
- Scheduling a callback alone does not cancel Operations.
- An Operation describes work, not completed output. Resolve it before tool-result conversion, size checks, display, message storage, or provider serialization.
- Do not add a base class, status framework, start callback, implicit registration, nested Operation support, or ToolResult subclass.

Limits to state explicitly:

- Startup remains ordinary Python and cannot be forcibly interrupted. Keep it short; use bounded poll calls for ongoing work.
- Plugin must clean up resources if it raises after creation but before returning Operation.
- Plugin owns cancellation correctness, descendants, resource cleanup, and honoring poll deadlines. A hung tool, blocking poll, or ineffective cancel delays draining.
- Cancellation cannot undo file writes or remote side effects.

## Lifecycle baseline

Use one small per-invocation record and small per-tool execution records. Reuse existing batch result structures rather than building a second registry/history.

Run owns cancellation signal, model thread, and current tool executions. Tool record owns thread, published Operation, cancel-claimed state, completion, and result/error. Record identity distinguishes ownership; raw tool-call ID is only display/provider metadata.

Register records before starting workers. A worker remains tracked until it actually exits and the coordinator joins it, even when stopped or when truncate removes its display message. Retain completed outcomes only as long as their associated batch/message requires them.

A stop request remains set on its run. Reject invoke while another run is executing/draining, before prompt transformation or message append. Never reset cancellation seen by an old worker.

Keep busy true through draining. Do not join workers, call providers/hooks, execute cancel functions, or dispatch scheduled callbacks while holding lifecycle/message locks.

### Stop API and Operation dispatch

Introduce one stop-request API and route `/stop`, Ctrl-X, and clear through it. Remove direct writable shared stop flags; adjust existing plugin checks to read the owning cancellation state where they wait.

Use the invocation coordinator to dispatch cancellation, not UI thread or an untracked cancellation thread:

1. Tool worker calls plugin function and publishes returned Operation under a short execution lock.
2. Coordinator polls all batch executions while joining with bounded waits, rather than waiting for the first tool and ignoring siblings.
3. For a canceled run, atomically claim each available, unfinished Operation's cancellation once.
4. Invoke claimed cancel functions outside locks.
5. Worker repeatedly calls Operation.poll with a short interval (e.g. 0.1 seconds) until result is not None, then stores outcome; worker completion is recorded in finally. Do not spin up another thread for each poll call.
6. Continue until all started workers have really exited; join before run cleanup.

Cancellation during startup is not lost: canceled run stays canceled, and coordinator notices later publication. If publication is followed by immediate completion, no cancellation is needed for already-finished work.

Cancel exceptions must not abandon polling or other tools: report through `app.debug_print`, continue draining, and do not retry cancellation. A terminal poll failure uses ordinary tool-error handling; plugin must release its resource before raising, since core cannot infer underlying lifetime from an exception. These are genuine plugin failure paths, not reasons to silently detach work.

Check cancellation before starting each tool and before committing provider output, tool results, hooks, and another model turn. Stop during streaming must not launch tools from an earlier assigned LLMResult. Explicitly close provider iteration on early exit when supported, before dispatching transitions; do not implement async-provider cancellation here.

## Schedule and clear

Preserve existing schedule API and callback thread behavior:

- One pending callback; second call raises without replacing first.
- Schedule requests exit after current turn/batch and waits for siblings; it does not request cancellation.
- Stop does not cancel pending callback.
- Boundary means joined tool workers, closed provider iteration, completed hooks and old-run cleanup. Old model-loop thread may dispatch callback and return, but must make no old-run writes afterwards.
- Callback may synchronously clear/invoke or schedule another callback.
- Tool must not wait for its scheduled callback: callback waits for tool.

Serialize invoke admission, scheduling, stop, clear, and callback selection with a short lifecycle lock. Reserve transition ownership so an external invoke cannot enter between old cleanup and callback clear/invoke. Callback executes outside locks; allow synchronous calls from its owning thread. Reservation release must only release that reservation, never reset a new run started by callback. Idle callback exceptions must also release ownership.

Clear semantics remain an outstanding user decision. In discussion, user requested clarification rather than approving the proposed rule. Do not silently interpret this plan as approval. Before implementation, confirm:

- Clear cancels pending callback, requests stop, and defers reset until draining finishes.
- Repeated clears coalesce.
- Schedule rejects while clear is pending.
- Idle clear remains synchronous.
- A callback already claimed is executing, not pending; external clear must not race its transition. Simplest policy is rejecting external clear during claimed dispatch, while allowing callback's own clear.

If user chooses another rule, update this section before coding. Do not add a general callback queue.

## History and UI consistency

Completed tool batches must commit atomically under message lock after checking owning run, cancellation, and batch validity. Coordinate this check with stop/clear through consistent short lock ordering.

If stopped batch is discarded, remove its assistant tool-call message and associated rows; preserve earlier valid batches and latest user prompt. Do not leave dangling tool calls in next provider request. Truncate must invalidate the specific removed batch, not resurrect it when siblings finish. Live execution records remain tracked independently of removed display messages.

Render executing vs completed/error from execution outcome, not merely missing tool result. A canceled but live worker is still executing/draining. Do not invent an unbounded outcome history keyed only by tool ID.

Question/approval waits must capture owning cancellation signal and remove only their own draw function on answer or cancellation. Replace `ui_stack.pop()` in those paths. Canceled approval must return denial, never None. Include escalation's analogous blocking UI path so stop does not hang draining.

Some plugins call approve directly before returning an Operation. Keep that synchronous helper contract; it cannot itself return Operation without changing its callers. Do not migrate all UI code into operations for this task.

Fork copies conversation/configuration, not run records, locks, pending callbacks/clear, transitions, cancellation, current output, suspension, or tool executions.

## Built-in shell tools

After core lifecycle works, migrate bash/powershell to demonstrate Operation without adding process-specific core APIs:

- Keep approval, executable resolution, public parameters, combined output formatting, exit-code prefix, and timeout result.
- Use Popen, return Operation with quick cancel and bounded poll closures.
- Share a small plugin-local subprocess helper if it removes existing duplicated code.
- Poll deadline and command timeout are separate: exhausting poll's wait returns None; exceeding command timeout requests termination. Continue polling until resources are reaped, then return timeout error.
- Collect stdout/stderr without filling pipes and blocking the child. A bounded communicate(timeout=wait) adapter may serve positive waits; verify its zero-timeout behavior rather than assuming communicate(timeout=0) implements an instant completed-result check. Keep adapter state/output between polls, without losing or duplicating output.
- Stop and timeout cancellation must safely handle already-exited processes.
- Cancel subprocess tree, not only shell. Inspect existing platform helpers first. On Windows use an appropriate tree-termination mechanism; on POSIX create/terminate a process group. Keep platform details in plugin.
- No blocking UI waits in cancel and no process handles stored in Context history.

If reliable process-tree handling pushes production changes past budget, discuss splitting shell migration into a follow-up. Do not claim shell cancellation works if only API exists and built-ins still use subprocess.run.

Do not expand into subagent orchestration or network-provider overhaul. Existing explore_agent parent tool remains tracked until child wait and removal finish; stop propagation to child is optional follow-up.

## Implementation order

1. Confirm clear/transition policy; inspect actual definitions and current working tree.
2. Add Operation value and concise purpose/contract documentation.
3. Fix run ownership and honest tool draining; integrate Operation publication, bounded polling, cancellation dispatch, and error handling.
4. Serialize invocation/transition admission and defer clear safely.
5. Fix atomic history commit, truncate invalidation, UI waits/rendering, and fork resets.
6. Migrate shell tools if within agreed scope/budget.
7. Update existing tests, add focused event-driven coverage, run entire suite, inspect full diff.

Do not implement a scheduler, reaper, executor framework, operation inheritance hierarchy, forced thread killing, or arbitrary tool isolation.

## Tests

Use Events/barriers with bounded waits and finally blocks releasing test workers. Avoid timing-only proofs and real indefinitely hung threads.

Operation coverage:

- Poll returns None until completion, then string or ToolResult including attachments; cancel never called. Empty completed results are not mistaken for pending. Plain tool results unchanged.
- poll(0) returns immediately for both pending and finished work; positive wait returns on completion or deadline without restarting/canceling work. Core uses bounded intervals rather than a five-second default that delays cancellation.
- Stop during polling calls cancel once; a cancel that merely signals a request does not mark tool finished. Context and tracking remain busy until poll reports actual completion and worker exits.
- Stop during startup is remembered; returned Operation is canceled then drained.
- Parallel Operations: one blocked sibling cannot prevent cancellation dispatch to others.
- Terminal poll failure after plugin resource cleanup and cancel failure do not leak tracking; cancel failure does not end polling or release context early. Error reporting uses debug_print.
- Schedule alone does not cancel; stop plus schedule cancels work but callback still waits for all workers.

Lifecycle/history coverage:

- Stop without pending callback still joins live tools.
- Invoke while draining rejected before transform/message append; old cancellation stays set.
- Schedule before stop/during drain/after completed run observes same safe boundary.
- Repeated scheduling preserves first callback; nested idle schedule, callback exception, and callback-started run release only own transition state.
- Concurrent invoke rejected between cleanup and callback completion; clear can cancel callback before claim. Claimed callback follows agreed clear policy.
- Handoff waits for sibling late Context writes, clears them, then starts exactly one clean run.
- Deferred clear and repeated clear honor agreed policy; stop/clear cannot interleave half a tool-result batch.
- Stop during stream does not launch previously seen tool calls.
- Discarded batch leaves valid provider history; completed earlier batches survive.
- Truncate while siblings run cannot resurrect removed calls/results or erase live tracking.
- Reused tool IDs cannot confuse execution ownership or completion cleanup.
- Finished tool and canceled-but-live sibling differ in bookkeeping/rendering.
- Two concurrent question/approval dialogs: answer/cancel removes only owning UI entry; canceled approval denies.
- Fork inherits no lifecycle state.

Shell coverage if migrated:

- Normal output, nonzero exit, timeout remain compatible.
- Repeated pending polls preserve combined output; zero-timeout checks work even when the process has exited but pipe output still needs collection.
- Real short-lived subprocess cancellation is reaped before callback; verify descendants are terminated using platform-appropriate tests.

Run `python -m unittest discover -s tests` (verify project test entry point first), then git diff/status. Preserve unrelated user edits.

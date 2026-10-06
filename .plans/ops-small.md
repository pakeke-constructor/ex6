# Small Operation implementation

## Motivation

ex6 should remain a thin harness. Current operations diff is too complicated to
merge: it combines a small tool-return API with lifecycle, history, and UI work.
Reduce this to the smallest implementation of cancellable tool work.

## Core simplification

- Tool worker owns its Operation completely: call tool, then loop over poll(0.1).
- Between polls, if ctx.stop_early and not locally canceled, call cancel once.
  Catch/log cancel failure and continue polling. Startup cancellation is naturally
  remembered by the same flag. No concurrent poll/cancel from core.
- Delete _ToolExecution, publication locks, cancellation dispatch, and execution
  registry changes. Keep existing threads, results, and _active_tools unchanged.
- Coordinator only joins workers; remove early return that abandons live threads.
- Keep plain ctx.stop_early and original /stop and Ctrl-X assignments.
- Guard invoke admission before prompt transformation using existing _msg_lock.
  Reject while busy; keep busy through worker joins.
- Keep minimal pending-clear flag: clear requests stop, cancels pending schedule,
  and resets after workers join. No new lifecycle lock or transition framework.
- Keep streaming stop check before launching tools and close provider iterator.
- Do not include render bookkeeping, fork overhaul, or atomic-history redesign.
  Retain only resets required by newly introduced state.
- Discard stopped assistant tool-call batch so next invocation has valid history.

## PowerShell

- Keep implementation inside tools.py; no process helper module.
- Windows-only implementation, matching existing public tool description.
- Popen, two temporary output files, deadline, taskkill /T /F on cancellation.
- Poll waits boundedly for PowerShell and taskkill, reads output after termination,
  closes files, and returns existing output/exit-code/timeout format.
- Temporary files avoid pipe buffering deadlocks and special communicate(0) logic.
- No POSIX support branch, cancel lock, or repeated-result cache; core polls only
  until completion and serializes cancellation with polling.
- Keep repetition guard from stringifying Operation; defer guard support for its
  completed output.

## UI decision

Keep existing stack with small stop checks in blocking dialogs and remove only
own draw function. Canceled approval denies. No UI lifecycle abstraction.
Explore/websearch remain ordinary threads and finish naturally.

## Tests and budget

- Restore existing handoff tests except actual handoff ready=True correction and
  smallest adjustment for deferred clear.
- Replace sprawling OperationTests with focused tests: result conversion (including
  empty result), cancel once plus honest draining/invoke rejection, and PowerShell
  output/timeout or cancellation smoke coverage.
- Reuse existing handoff sibling-draining test; do not duplicate it.
- Aim for under 100 added production lines and about 60–90 new test lines.
- Run entire suite, inspect diff and line counts. Stop if reduction requires
  weakening worker draining or claiming unimplemented process-tree cancellation.

## Relevant files

- ex6.py: Operation, call_tools, Context.invoke/clear
- _ex6/tools.py: powershell, blocking dialogs, guard_repeat_calls
- _ex6/commands.py: restore original stop flag assignment
- tests/test_handoff.py and tests/test_operations.py

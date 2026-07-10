# Priority changes

- Fix `edit_file_lines(end=0)` insertion path; do not validate line 0.
- Fix `edit_file_lines(end=0)` validation, sometimes the LLMs overwrite the wrong lines. Remember, when edit_file_lines is called for lines X through Y, every line after line X gets invalidated.
- Ensure `Context.invoke.run()` always clears `llm_is_running` / `llm_suspended` in `finally`.
- Add approval to powershell and bash calls
- Make `ctx.messages` access safe: mutate via helper/lock; render from snapshot.
- Enforce invariant: only UI thread mutates UI stacks/panels.
- Enforce invariant: all `ctx.messages` mutations go through one locked path.
- Enforce invariant: mutating tools are serialized or explicitly marked safe for concurrency.
- Return tool error messages for unknown tool calls instead of silently ignoring.
- Make gitignore handling per `ctx.cwd`, not import-time process cwd.
- Update README: `_ex6` plugin folder, current features, remove stale cut-off text.
- Add tiny tests for command dispatch, type/schema conversion, `edit_file_lines`, plugin loading order.


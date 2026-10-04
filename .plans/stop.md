# Immediate /stop

Motivation: `/stop` should release the current context promptly, including while waiting for the first token or a stalled stream. Preserve ex6's explicit, plugin-controlled provider interface; no forced thread killing or hidden worker activity modifying a later invocation.

Current behavior:
- `_ex6/commands.py:stop` and Ctrl-X set `ctx.stop_early`.
- `ex6.py:Context.invoke` observes that flag only after the provider yields.
- Both providers use blocking SDK requests/stream reads. Closing a synchronous stream from another thread is not a reliable immediate-cancellation guarantee, and no stream exists while waiting for response headers.
- Tool waits already poll stop, but their threads continue running.

Recommended scope: prompt local cancellation of LLM requests, not force-termination of arbitrary tools or a guarantee that the remote server stops generating/billing.

Plan:
1. Add one Context stop entry point; route `/stop` and Ctrl-X through it. Integrate clear with cancellation rather than merely marking a still-active worker idle.
2. Give each invocation its own cancellation state/identity. Prevent canceled workers from appending output, starting tools, or resetting a newer invocation's state.
3. Use cancellable async SDK tasks inside the OpenRouter and Codex providers, behind the existing synchronous generator interface. A small async-to-generator bridge owns task/loop cleanup; stop schedules cancellation without blocking the UI.
4. Handle cancellation as normal termination, not an API error. Clean up streams/clients in finally. Check cancellation before retrying Codex authentication; synchronous token refresh remains a caveat unless also converted.
5. `/stop` early must truncate to the most recent valid message/tool-call boundary: no dangling/incomplete messages or dangling tool calls. Discard in-progress assistant output and clear its UI state. Preserve the latest user prompt and completed turns. Treat an assistant tool-call message plus all matching tool-result messages as one atomic history batch: if any result is missing, remove the entire batch, including any results already appended. Preserve completed batches. Do not fabricate cancellation results or retain partial tool-call arguments. Perform rollback under the message lock, clean removed tool rows, and prevent canceled workers from writing after rollback. Rollback changes history only; it cannot undo tool side effects.
6. Test cancellation before response headers, during a stalled stream, immediately followed by another invocation, and around completion/tool transitions. Include parallel tool calls with only some results complete, stop racing with tool-result commit, preservation of completed batches/latest user prompt, and removal of partial output. Check diff and run tests.

Relevant files:
- `ex6.py`: Context.invoke, Context.clear, call_tools, Ctrl-X handler.
- `_ex6/commands.py`: stop.
- `_ex6/provider.py`: invoke_llm.
- `_ex6/provider_openai.py`: invoke_llm, _codex_client, token refresh.

Rough estimate, subject to SDK cancellation verification:
- Best-effort synchronous stream-close approach: 40–80 added production lines, low-to-medium complexity; insufficient for reliable cancellation before headers or stalled synchronous reads. Not recommended if immediate means dependable.
- Recommended cancellable provider path, including valid-history rollback: 140–210 added production lines plus roughly 60–90 test lines; medium complexity. Existing provider loops also need restructuring, so changed lines exceed added lines. Rollback adds roughly 20–30 production lines to the original estimate.
- Force-stopping subprocesses/arbitrary tools is separate scope. Python threads cannot safely be force-killed; do not include it in this change.

First implementation check: prototype async cancellation against installed SDK, covering both request setup and stream iteration, before restructuring both providers. If total new code requires more than 300 lines, stop and simplify scope with user.

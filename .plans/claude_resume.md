# claude_resume: stop claude backend flattening history on divergence
Status: active

## Motivation
User: "The new top priority is to mitigate the effects of this."
provider_anthropic runs a long-lived `claude -p` process per ctx and only sends deltas. `new_messages()` requires ex6 history to be a strict identity-extension of `process.seen`; otherwise the process restarts and `_bootstrap_prompt` flattens the whole history into ONE user message ("Continue this conversation... [assistant tool calls] {json} [tool] ...").
Cost: model reads its own past as quoted text (weak ownership, may imitate text tool-call format), tool results gain "context, not instructions" framing, tool-result attachments dropped, full cache miss. Every subsequent restart re-flattens everything.

Triggers (all common):
- pop (`_pop_prompts` -> `ctx.truncate`), unless back to first prompt
- stop mid-turn (assistant reply dropped -> `new[0].role != "assistant"`)
- fork (new ctx, no process)
- ex6 restart (processes not persisted)
- `/purge` (`purge_cache`) and worktree switch (copy Message objects -> `a is not b` fails)
- changed system prompt / tools / native tools / cwd
Fine: clear, handoff (single user msg sent raw).

Also blocks `.plans/purge.md` (agent checkpoint/purge would make divergence frequent).

## Directions (investigate, pick simplest that works)
1. Real resume: drop `--no-session-persistence`, synthesize a session JSONL in claude's format from ex6 history, start with `--resume <id>`. Gives true structured history for every trigger. Risk: undocumented format/protocol drift -> needs a check in `providers_test.py`.
2. Check whether `--input-format stream-json` accepts replayed assistant/tool_result events (would avoid files entirely).
3. Cheap wins regardless: compare messages by content not identity (fixes `/purge`, worktree when content unchanged); maybe keep partial assistant msg on stop.
4. Fallback: improve flattened bootstrap framing.

## Files
- `_ex6/provider_anthropic.py`: `_bootstrap_prompt` (~198), `_ClaudeProcess` args (`--no-session-persistence` ~257), `new_messages` (~358), `invoke_anthropic` (~534), `_processes`
- `_ex6/providers_test.py`: smoke tests for silent provider failures
- `ex6.py`: `truncate` (~1417), `purge_cache` (~1187), `fork` (~1494), `_pop_prompts` (~2757)

# Master Prompt: Enable Native Web Search in Our Custom Harness (Claude + Codex backends)

## Overarching motivation

We run our own custom agent harness. It drives `claude -p` (and Codex via `codex exec`) as backends, legitimately, and passes in **our own custom tools**. Every built-in tool (Bash, Edit, Write, etc.) is currently disabled on purpose; the agent should only act through our tools.

The goal: **give the agent live web search by re-enabling each backend's native search tool, while keeping everything else locked down.** We are not building or hosting our own search, and we are not nesting `claude -p` calls inside a custom search tool. The backends already ship server-side search; we just stop disabling it.

Why this approach:
- **Highest ROI, lowest effort.** It's a flag and config change, not a new subsystem.
- **It uses the vendors' maintained search backends.** Anthropic's server-side `web_search` and OpenAI's hosted search tool.
- **It keeps one agent session.** No spawning sub-sessions per lookup, so it stays faster and cheaper.

Context on policy: Anthropic now bills programmatic `claude -p` / Agent SDK usage from a separate programmatic budget, not interactive subscription limits. Our use is legitimate; just be aware that searches cost money from that budget.

## In my own words (the human)

> "I am currently hooking into `claude -p` legitimately to run my own custom harness."

> "I already have `claude -p` running with my own custom tools that I pass to it. I disable every tool directly; including bash, edit file, etc."

> "Can I just... not disable websearch or something?"

> "What about for codex?"

Yes to both. Your job is to make that happen cleanly.

## Objectives

1. **Claude backend:** Allow the built-in `WebSearch` tool in our `claude -p` invocation. Keep every other built-in disabled.
2. **Codex backend:** Explicitly set the web search mode in our `codex exec` invocation. Don't rely on defaults.
3. **Make it configurable.** Use one harness-level setting (e.g. `web_search: off | search | search+fetch`, or `cached | live` for Codex) so it can be toggled per run or per agent without code edits.
4. **Verify it works** end-to-end on both backends with evidence from the event streams.
5. **Do not widen the attack surface** beyond what's intended (see Security).

## Implementation details

### Claude (`claude -p`)

Today we likely pass something like `--tools ""` or a `--disallowedTools` list. Change it so `WebSearch` is the only enabled built-in:

```
claude -p "<prompt>" \
  --tools "WebSearch" \
  --allowedTools "WebSearch" \
  <existing MCP / custom tool flags unchanged>
```

- `--tools` controls which built-in tools exist. `--allowedTools` pre-approves them so headless mode never stalls on a permission prompt.
- If we disable tools via `--disallowedTools` instead, remove `WebSearch` from that list.
- **Check the flags against `claude --help` on our installed version** before relying on them; flag names change between versions.
- `WebFetch` (reading full page content) is a **separate** tool. Leave it **off by default**. Expose it only behind the `search+fetch` setting.

How it works, so you know what to expect:
- `WebSearch` runs a small secondary model that calls Anthropic's server-side `web_search` tool, which can do up to about 8 searches per call.
- It returns titles, URLs and a short summary, not full pages.
- It only works with first-party Anthropic auth. On Bedrock or Vertex the tool is silently hidden.

### Codex (`codex exec`)

Codex search is OpenAI's hosted tool. Set it explicitly per run:

```
codex exec -c web_search="cached" "<prompt>" # default; OpenAI's pre-indexed cache
codex exec -c web_search="live" "<prompt>" # real-time results
codex exec -c web_search="disabled" "<prompt>"
```

- Prefer the `-c` override over `--search`, since `exec` may not accept every top-level flag. Confirm with `codex exec --help`.
- A dedicated `--profile` with `web_search = "live"` in config is an acceptable alternative.
- **Pin the mode explicitly.** `--yolo` or any full-access sandbox silently upgrades search to `live`.
- In live mode the model can open and read pages within the search tool itself. There is no separate fetch toggle, so "live" is roughly equivalent to Claude's `search+fetch`.
- Search requires OpenAI-hosted models. A custom `model_provider` will not have it.

## Security constraints (non-negotiable)

- **Web content is untrusted input.** It shares a context with our custom tools, so a malicious page can attempt prompt injection to trigger those tools.
- Default to the narrowest setting: Claude `WebSearch` only, Codex `cached`.
- Enabling page reading (Claude `WebFetch`, Codex `live`) must be an explicit, per-run opt-in.
- Audit our custom tools for anything with side effects (writes, sends, payments, deletes, external calls). Write a short note listing which tools are dangerous to combine with page reading, and recommend mitigations, such as confirmation gates or disabling those tools when fetch is on.
- Do not re-enable Bash, Edit, Write or any other built-in as part of this work.

## Verification / acceptance criteria

- [ ] **Claude:** run with `--output-format stream-json --verbose` and a prompt that requires current info, e.g. "What's today's top headline on <news site>?" The stream shows a `WebSearch` tool_use, and the answer cites real URLs.
- [ ] **Claude:** in the same run, confirm no other built-in tools are available. For example, ask it to run a shell command; it should be unable to.
- [ ] **Codex:** run `codex exec --json` with the same kind of prompt. The stream shows `web_search` items.
- [ ] **Codex:** confirm the mode matches what was configured. Test that `cached` doesn't silently become `live` under our sandbox settings.
- [ ] The harness setting toggles both backends correctly: off, narrow and wide.
- [ ] Our custom tools still work exactly as before on both backends.
- [ ] Log or record search usage counts per run so we can track cost.

## Deliverables

1. Code changes to the harness's backend invocation (Claude + Codex), behind a single config setting.
2. A short README section covering how to toggle search, what each level enables, and the cost and security implications.
3. The security note on custom tools that are risky alongside page reading.
4. Verification evidence: trimmed stream excerpts showing search tool calls on both backends.

## Working style

- Bias to action. This should be small, so ship the narrow version first, verify it, then add the toggle and the wider modes.
- If a flag doesn't exist or behaves differently on our installed CLI versions, check `--help`, adapt, and note the discrepancy. Don't guess.
- Don't build a custom search integration (Exa, Tavily, etc.) unless native search proves unworkable. If it does, stop and report why before building anything.
# Native web search verification

Motivation: expose vendor-native search without re-enabling built-in shell/file
tools or adding another search agent. Scope: Anthropic and OpenAI only.

## Implementation / version discrepancy

Checked `claude --help` (2.1.310) and `codex exec --help` (0.164.0).
Claude supports `--tools` and `--allowedTools`. Existing MCP transport, custom
system prompt, and permission flags stay unchanged. Native results are handled
inside Claude; only custom calls go through ex6's tool dispatcher. Unknown
self-resolved custom-tool results still fail validation.

Contrary to the original plan, `_ex6/provider_openai.py` does NOT invoke
`codex exec`: it calls the Codex Responses endpoint directly. Retained that
backend and added `{"type":"web_search","external_web_access":false|true}`
to its existing tool list. Off omits that item. No new CLI backend or sandbox.
OpenRouter invocation is unchanged; `provider.py` hosts shared settings/logging.

## Tests (2026-10-27)

Temporary Python smoke scripts invoked real `Context.invoke()` with one harmless
custom `smoke_echo` tool. All six backend/mode combinations passed, including
custom-tool execution and continuation. Scripts and raw captures were removed
after inspection; trimmed evidence is below.

- Claude model: `anthropic/claude-sonnet-4.6`.
- OpenAI model: `openai/gpt-6-luna`. Initial `gpt-5.3-codex` attempt returned
  `model_not_found` for this account; switched to an available model.
- Off: no native calls on either provider; custom echo succeeded.
- Search: Claude 1 search / 0 fetches; OpenAI 2 searches / 0 fetches.
- Search+fetch: Claude 1 search / 1 fetch; OpenAI 2 searches / 2 page opens.
- Claude init tool inventories were exactly `[smoke_echo]`,
  `[WebSearch, smoke_echo]`, and `[WebFetch, WebSearch, smoke_echo]`.
- Narrow-mode prompts also requested built-in shell execution; both models
  reported no shell tool. No shell tool was exposed or executed.
- Captured the actual OpenAI HTTP request body, not just SDK kwargs: it retained
  `external_web_access:false` through custom-tool continuation. Local assertions
  checked both `yolo=False` and `yolo=True`; neither changes the mapping.
- Local assertions covered default/env/context precedence, invalid settings,
  idle-only command changes, per-user-run counter accumulation/reset, native
  result routing, unknown-result rejection, and Claude process invalidation
  when the mode changes. Python compilation passed.

Prompt (narrow): search for latest Python release on python.org, cite a URL,
try `echo NATIVE_SHELL_TEST` using a built-in shell only, then call `smoke_echo`.
Wide prompt additionally requested opening https://www.python.org/downloads/.

Claude stream excerpts (unrelated fields omitted):

```json
{"type":"system","subtype":"init","tools":["WebSearch","smoke_echo"]}
{"type":"tool_use","id":"toolu_01MTkrXLcnPxfFDGQuV6rJSZ","name":"WebSearch","input":{"query":"latest Python release 2025 python.org"}}
{"type":"tool_result","tool_use_id":"toolu_01MTkrXLcnPxfFDGQuV6rJSZ","content":"Web search results for query: ... Links: ... https://www.python.org/downloads/ ..."}
```

The query's year was model-generated. Returned results and the final answer
reported Python 3.15.2. Final answer cited `https://www.python.org/downloads/`
and `https://www.python.org/downloads/source/`, reported no built-in shell,
and included `SMOKE_ECHO_OK: native-web-test`.

Wide-mode Claude returned a WebFetch result: "The latest Python release on this
page is **Python 3.15.2**. The page offers it as the 'latest source release' and
lists it first in the release table."

OpenAI wire tools (custom function schema omitted):

```json
[{"type":"function","name":"smoke_echo"},{"type":"web_search","external_web_access":false}]
```

OpenAI narrow stream excerpt:

```json
{"type":"response.output_item.done","item":{"id":"ws_0fa79ecc159b0268016ae02ca8569887d091ea6b8ec14110d3","type":"web_search_call","status":"completed","action":{"type":"search","query":"site:python.org/downloads/ latest Python release","queries":["site:python.org/downloads/ latest Python release"]}}}
```

OpenAI wide-mode stream excerpt (wire flag was `true`):

```json
{"type":"response.output_item.done","item":{"id":"ws_071fcebedb0da67f016ae02cac6de087d083356f3b87450ecc","type":"web_search_call","status":"completed","action":{"type":"open_page","url":"https://www.python.org/downloads/"}}}
```

Final OpenAI wide answer: "Python **3.15.2** is the latest release; python.org
lists it dated October 25, 2026. ([python.org](https://www.python.org/downloads/))
Smoke test result: `SMOKE_ECHO_OK: wide-test`."

## Direct context-window inspection / limitations

Inspected `ctx.get_messages()`, Claude's assistant/user tool blocks, and actual
OpenAI serialized request bodies. Did not rely solely on model claims.

The ex6 message history for each smoke run was:
1. User request.
2. Assistant text (sometimes empty) plus `smoke_echo` custom tool call.
3. Custom tool result `SMOKE_ECHO_OK: ...`.
4. Final assistant answer with source links.

Native search calls/results are NOT ex6 `Message` tool rows. They execute inside
the vendor. We retain emitted native events in `native_web` logs/volatile data
and show call activity as reasoning chunks; this is not full native-context
synchronization with ex6's conversation display.

Claude's live process retains the search-result text, links, and source-citation
reminder in its internal conversation. These were directly visible in its
streamed `user` tool-result blocks. Restart/fork/bootstrap replays only ex6
messages, so native-only context is lost. Existing CLI environment/model
reminders also remain; enabling search does not remove them.

OpenAI emits search action metadata and answer citations, not full search-result
page text. Its follow-up HTTP input contained user prompt, custom function call,
and custom function output, but no hosted search items. With existing stateless
history conversion this caused repeat searches after `smoke_echo` (2 searches
instead of 1; wide mode also repeated the page open). No hidden persistence or
extra injected prompts were added to conceal this limitation. Preserving full
native context across turns would require a separate history/replay change.
Cache-only prevents fresh external retrieval; it does not promise that cached
page content cannot be read. We cannot inspect the hosted service's internal
context or independently audit its cache implementation.

## Custom-tool security audit

Search summaries and cached pages can inject instructions too. Page reading
increases exposure; it is not safe merely because built-in shell is disabled.

- `_ex6/tools.py`: `write_file`, `edit_file`, `edit_file_codemode`,
  `edit_file_lines`, `patch_file` modify files. `bash` and `powershell` can
  execute code, delete files, send data, install software, or invoke payment
  APIs. Existing `_approve` / `_approve_command` gates remain; `yolo` or an
  allow-all approval handler bypasses them. Review commands and diffs, not just
  the model's justification. Disable these tools on page-reading agents where
  practical.
- `_ex6/tasks.py`: `plan_write`, `plan_add_log`, `plan_done` write persistent
  plans without the generic file-write approval UI. Disable them for research
  agents, or add project-specific confirmation before trusting web-derived
  plan changes (plans can later drive autonomous work).
- `_ex6/agents_sound.py`: `freesound_search` makes authenticated external
  requests; `freesound_download_preview`, `freesound_install`, and
  `freesound_discard_preview` download/write/delete assets and update credits.
  Mutation tools have approval gates. Keep them gated or absent during browsing.
- `_ex6/web_tools.py` / `_ex6/tools.py`: `websearch_agent` and `explore_agent`
  launch additional model work, consume quota, and disclose their prompts to a
  provider. `handoff` changes conversation continuity. Treat web-derived
  delegation instructions as untrusted, cap spend, and do not pass secrets.
- Local read tools can expose sensitive content to the model. Existing secret
  filename/gitignore checks are not a complete data-loss prevention boundary.
  Native search queries and fetch URLs can then exfiltrate that content without
  going through custom-tool approval. Use isolated, read-only research contexts
  with no secrets and review what is transferred back to a privileged agent.

No dedicated send/payment tool was identified in this audit; unrestricted shell
and project plugins can provide equivalent capabilities. Audit project-specific
tools separately. Native web counts are observability, not budget enforcement.

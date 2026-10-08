# `claude -p` for a custom harness: empirical findings

Context for an agent picking this up: the user wants to build their own agent harness on `claude -p` (Claude Code headless mode) with custom system prompts and custom tools. They care about (1) prompt caching and (2) exactly what goes into the context window.

**Method.** Findings come from Claude Code **v2.1.293** on Linux, 2026-10-08. `ANTHROPIC_BASE_URL` was pointed at a local mock Messages API (`mock.py`, at the bottom of this file) that logs every request body. Runs used `env -i` with a fresh empty `HOME` to approximate a clean machine. Token counts are estimates (chars / 3.6).

**Caveats:**
- The mock could not fetch server-side feature flags. On a real machine the default tool set and some injected text may differ by version or flag.
- Cache hits were *not* verified against the real API. Only byte-stability of the request prefix was checked.
- A first run inside a heavily customised cloud sandbox produced extra content (46 tools, a user-email reminder, a "Saving skills" section). All of it came from that sandbox's env vars. Ignore it; on a real machine, check your own env for `CLAUDE_*` vars.

---

## TL;DR

- **Use `--bare`.** Without it, `--system-prompt` still ships ~18k tokens of built-in tools, a ~2.4k-token `# Environment` message, and reminders.
- The tightest configuration:
  ```bash
  claude -p --bare --system-prompt-file sp.txt --tools "" \
    --mcp-config tools.json --strict-mcp-config \
    --output-format stream-json --verbose < /dev/null
  ```
  - Context = 2 fixed preamble blocks (~38 tok) + your system prompt + your MCP tool schemas (verbatim) + messages. Nothing else.
- **Caching works automatically** and the prefix is byte-stable across tool-loop turns and across `--resume` processes.
  - Default TTL is 5 min with an API key. Set `CLAUDE_CODE_PROMPT_CACHE_TTL=1h` for slow loops (1h writes cost more).
- You **can** control exactly which tools exist. You **cannot** edit built-in tool descriptions or schemas; reimplement them as MCP tools if wording matters.

---

## 1. What is in the request

### Always present, even in the cleanest mode

| Location | Content | Removable? |
|---|---|---|
| `system[0]` (uncached) | `x-anthropic-billing-header: cc_version=2.1.293.72d; cc_entrypoint=sdk-cli;` | Yes: `CLAUDE_CODE_ATTRIBUTION_HEADER=0`. Static per CLI version. |
| `system[1]` (cached) | `You are a Claude agent, built on Anthropic's Claude Agent SDK.` (~17 tok) | No flag found. With `--append-system-prompt` it is instead `You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK.` |
| `system[2]` (cached) | Your `--system-prompt` text, verbatim | — |

Other request params the user does not set by default:

| Param | Value |
|---|---|
| `max_tokens` | `128000` |
| `thinking` | `{"type":"adaptive","display":"omitted"}` |
| `context_management` | `{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}` |
| `output_config` | `{"effort":"medium"}`; set with `--effort` |
| `metadata.user_id` | JSON containing device_id, account_uuid, session_id |
| `stream` | `true` |

Multiple `anthropic-beta` headers are also sent, e.g. `prompt-caching-scope-…`, `mid-conversation-system-…`, `per-turn-control-…`. A per-turn `{"role":"system","content":[],"output_config":{…}}` message also appears in history.

### By configuration (clean env)

| Config | System prompt | Tools | Injected messages |
|---|---|---|---|
| `claude -p` (default) | ~1.6k tok Claude Code prompt | 23 tools, ~18.4k tok | Git attribution reminder (~160 tok) + `# Environment` system message (~2.4k tok). May also add an auto-memory reminder (~650 tok) and CLAUDE.md. |
| `--system-prompt X` (not bare) | X | **Still 23 tools, ~18.4k tok** | **Still** the attribution reminder and the Environment block |
| `--bare` (no system prompt) | `CWD: …\nDate: YYYY-MM-DD` (changes daily, so the cache breaks daily) | Bash, Edit, Read (~0.9k tok) | none |
| `--bare --system-prompt X` | X | Bash, Edit, Read | none |
| `--bare --system-prompt X --tools ""` | X | none | none |

Default tool list seen (non-bare, clean env): `Agent, Bash, CronCreate, CronDelete, CronList, DesignSync, Edit, EnterWorktree, ExitWorktree, ListAgents, Monitor, NotebookEdit, PushNotification, Read, ReportFindings, ScheduleWakeup, SendMessage, Skill, TaskStop, WebFetch, WebSearch, Workflow, Write`.

### The `# Environment` message (non-bare only)

It is injected as a system-role message after the first user message. It contains:
- cwd, whether the directory is a git repo, platform, shell, OS version
- security boilerplate about downloaded files
- model name, model ID and knowledge cutoff
- the subagent type list
- MCP server `instructions` under `# MCP Server Instructions`
- today's date

Its size depends on the tools present: ~2.4k tok with all tools, ~256 tok with a 5-tool subset. The git attribution reminder only appears when Bash is present.

### What `--bare` skips

Bare mode skips hooks, skills, plugins, MCP auto-discovery, auto-memory, CLAUDE.md, LSP and system reminders. In bare mode:
- Claude isn't told when files change on disk.
- No skills list is injected.
- No background tasks run.
- MCP server `instructions` are **not** injected, so put anything important in your own system prompt.

Auth in bare mode is strictly `ANTHROPIC_API_KEY` or `apiKeyHelper` via `--settings`. It never reads OAuth or subscription credentials or the keychain.

---

## 2. Prompt caching

**Breakpoints are automatic:**
- `cache_control: ephemeral` sits on system blocks 1 and 2.
- Tools come before the system prompt in the cache prefix, so they are covered by that breakpoint.
- A rolling breakpoint sits on the latest message block(s); in tool loops it covers the last `tool_use` and `tool_result`.

**Stability was verified as follows:**
- Within a tool loop: `system`, `tools` and all other params were byte-identical, and messages were append-only.
- Across processes with `--resume <session_id>`: `system`, `tools`, `metadata` and all other params were identical, and history was replayed.
  - Minor difference: earlier user turns were re-serialised as a plain string instead of a single `[{"type":"text"}]` block. These should render identically on the API side, but this is unverified.
- Separate `claude -p` processes share the cache, because it lives on the server and is keyed by prefix.

**TTL:**
- 5 min by default with an API key.
- Override with `CLAUDE_CODE_PROMPT_CACHE_TTL=5m|1h` (v2.1.242+), which takes precedence over `ENABLE_PROMPT_CACHING_1H`. `FORCE_PROMPT_CACHING_5M` overrides it.
- 1h writes are billed higher.

**Things that invalidate the cache:**
- A CLI upgrade, which changes the billing header and possibly the tool schemas.
- Bare mode without `--system-prompt`, because the `Date:` line changes daily.
- In non-bare mode, the Environment block holds git status and the date. It sits after your first user message, so only the later part of the prefix churns.

**`--system-prompt-snapshot` is `on` by default.** The system prompt is recorded on the conversation's first request and reused verbatim on every later request and `--resume`, even if a later launch passes different text. Use `off` while iterating on prompt text.

`--exclude-dynamic-system-prompt-sections` moves per-machine sections into the first user message for cross-user cache reuse. It only applies with the default system prompt and is ignored with `--system-prompt`.

---

## 3. Tool control

| Flag | Effect on the `tools` array sent to the model |
|---|---|
| `--tools "A,B"` | Exact allowlist of built-ins; everything else is removed. |
| `--tools ""` | No built-ins. |
| `--allowedTools "X"` | **Permissions only.** Schemas are unchanged; it just auto-approves. |
| `--disallowedTools "Bash"` | Removes the tool from the request. |
| `--disallowedTools "Bash(rm *)"` | Tool stays in the request; matching calls are denied at runtime. |
| `--disallowedTools "mcp__srv__tool"` | Removes that MCP tool. |
| `--mcp-config <json-or-file>` | Adds MCP tools. |
| `--strict-mcp-config` | Ignores all MCP config except `--mcp-config` (bare already does this). |
| `--agents '{…}'` | Defines subagents. The Agent tool must also be in `--tools` (e.g. `--tools "Read,Agent"`). |

**Gotchas:**
- **Bare mode only has Bash, Edit and Read.** `--bare --tools "Read,Grep,Glob,WebFetch,Write"` silently yielded only `Read`. Non-bare honoured all five.
- **Unknown or unavailable names are dropped silently.** `TodoWrite` and `AskUserQuestion` vanished in `-p`.
- **Verify what loaded** using the `system/init` event in `--output-format stream-json --verbose`. It lists tools, `mcp_servers`, `mcp_server_errors`, `plugins` and `plugin_errors`.
- **MCP tools pass through verbatim**, except the name is prefixed `mcp__<server>__<tool>`. Example as sent:
  ```json
  {"name":"mcp__mine__get_weather","description":"Get the weather for a city.",
   "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}
  ```
  - Tools are sorted alphabetically, so the order is stable and cache-friendly.
  - Long descriptions may be truncated (`CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` exists; default unknown).
- **Built-in tool descriptions cannot be changed.** No flag exists. For full control over wording, use `--tools ""` and reimplement (e.g. your own `bash`, `read_file`) as MCP tools.
- **With `-p`, MCP startup waits** up to `MCP_TIMEOUT` (default 30s) for pending `--mcp-config` servers before the first turn. Invalid entries are skipped silently except in the `mcp_server_errors` field.

### Tool search / deferred loading

- **Activates with:** `ENABLE_TOOL_SEARCH=true` + `ToolSearch` included in `--tools` + 80 MCP tools, in non-bare mode.
  - The request then contained only `ToolSearch` (~415 tok) and a `DeferredToolPlaceholder` with `defer_loading: true`.
  - The 80 tool names were listed in the Environment message instead. Their full schemas would have cost ~9.1k tok.
- **Did not activate in these cases:**
  - Default auto settings with ~9k tok of tools (the threshold is presumably higher).
  - `--tools ""`, which also removes `ToolSearch`.
  - `--bare`, where it never activates.
- **Not tested:** whether loading a deferred tool mid-run preserves the cache. The mock can't do a real search round trip.

---

## 4. Other useful facts and flags

**Output and session flags:**

| Flag | Notes |
|---|---|
| `--output-format text` / `json` / `stream-json` | `json` gives `result`, `session_id`, usage and `total_cost_usd` (client-side estimate). `stream-json` needs `--verbose`; add `--include-partial-messages` for token deltas. |
| `--json-schema '<schema>'` | Output lands in `structured_output`. An invalid schema causes an error exit; the `format` keyword is not enforced. |
| `--resume <id>` / `--continue` | Resume by ID or the most recent session. `--resume` also accepts a transcript `.jsonl` path. |
| `--fork-session` | New session ID when resuming. |
| `--session-id <uuid>` | Use a specific session ID. |
| `--no-session-persistence` | Don't save the session to disk. |

**Model and run limits:**

| Flag | Notes |
|---|---|
| `--model` | Model alias or full name. |
| `--fallback-model` | Fallback when the primary is overloaded. |
| `--effort low\|medium\|high\|xhigh\|max` | Effort level. |
| `--max-budget-usd` | Spend cap for the run. |
| `--autocompact <auto\|tokens>` | Auto-compact window size. |
| `--betas` | Extra beta headers (API key users only). |

**Permissions:**
- `--permission-mode acceptEdits|auto|bypassPermissions|manual|dontAsk|plan`
- `--permission-prompts none` denies anything that would prompt and tells Claude not to retry.
- `--dangerously-skip-permissions` bypasses all permission checks.

**Context loading and input:**
- `--append-system-prompt[-file]`, `--settings`, `--add-dir`, `--plugin-dir`, `--setting-sources`
- `--input-format stream-json` for a long-lived process fed over stdin.

**Pitfalls:**
- Always redirect stdin (`< /dev/null`) or pipe input; otherwise each call waits 3s for stdin.
- Stdin is capped at 10MB.
- Without `--bare`, `-p` skips the workspace trust dialog and will run project `.claude/settings.json` hooks and `.mcp.json` servers in the working directory.
- SIGTERM exits with code 143 and leaves the turn unfinished; SIGINT ends the turn cleanly.
- Background subagents keep `-p` alive until done, with a 10 min idle cap (`CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS`).

**Relevant env vars:**

| Variable | Purpose |
|---|---|
| `CLAUDE_CODE_PROMPT_CACHE_TTL` | Cache TTL (`5m` or `1h`). |
| `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | Drop the billing header block. |
| `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` | Don't load CLAUDE.md files. |
| `CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1` | Remove git workflow instructions and the git status snapshot. |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS` | Max output tokens per request. |
| `CLAUDE_CODE_EFFORT_LEVEL` | Effort level; overrides `--effort`. |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | Auto-compact window in tokens. |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | Lower the auto-compact trigger percentage. |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | Disable updates, telemetry and other nonessential traffic. |
| `ENABLE_TOOL_SEARCH` | Turn on deferred tool loading. |

---

## 5. Strategic note

With `--bare --tools "" + MCP tools + custom system prompt`, Claude Code mostly provides four things:
- the agent loop
- MCP plumbing
- session persistence and resume
- permission handling and streaming events

The context overhead is only ~38 tokens. If the harness doesn't need those features, calling the Messages API directly (or using the Agent SDK in Python/TS) gives full control over the prefix, cache breakpoints, thinking and `max_tokens`.

Sources: https://code.claude.com/docs/en/headless · https://code.claude.com/docs/en/env-vars · `claude --help` (v2.1.293)

---

## Appendix A: `mock.py` (request-capturing mock API)

Usage:
```bash
python3 mock.py 9001 ./captures &
ANTHROPIC_BASE_URL=http://127.0.0.1:9001 ANTHROPIC_API_KEY=sk-test \
  claude -p hello --bare --system-prompt "X" < /dev/null
```
- Each request body is saved to `./captures/NNN_v1_messages.json`.
- Set `TOOLMODE=1` to make the first request return a `Bash` tool_use, which tests a tool-loop round trip.
- For a clean test, run under `env -i PATH=... HOME=<empty dir>`.

```python
import json, sys, os, itertools
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
counter = itertools.count()

def sse(ev, data):
    return f"event: {ev}\ndata: {json.dumps(data)}\n\n".encode()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("content-type","application/json"); self.end_headers()
        self.wfile.write(b"{}")
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n)
        i = next(counter)
        with open(f"{OUT}/{i:03d}_{self.path.strip('/').replace('/','_').split('?')[0]}.json", "w") as f:
            json.dump({"path": self.path, "headers": dict(self.headers), "body": json.loads(body or b"{}")}, f, indent=1)
        if "count_tokens" in self.path:
            self.send_response(200); self.send_header("content-type","application/json"); self.end_headers()
            self.wfile.write(b'{"input_tokens":10}'); return
        req = json.loads(body or b"{}")
        model = req.get("model", "x")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        w = self.wfile.write
        w(sse("message_start", {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":model,"content":[],"stop_reason":None,"stop_sequence":None,"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}))
        last = req["messages"][-1]["content"]
        has_result = isinstance(last, list) and any(b.get("type")=="tool_result" for b in last)
        if os.environ.get("TOOLMODE") and not has_result and req.get("tools"):
            w(sse("content_block_start", {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":f"toolu_{i}","name":"Bash","input":{}}}))
            w(sse("content_block_delta", {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":json.dumps({"command":"echo hi","description":"say hi"})}}))
            w(sse("content_block_stop", {"type":"content_block_stop","index":0}))
            w(sse("message_delta", {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":None},"usage":{"output_tokens":1}}))
        else:
            w(sse("content_block_start", {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}))
            w(sse("content_block_delta", {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}))
            w(sse("content_block_stop", {"type":"content_block_stop","index":0}))
            w(sse("message_delta", {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":None},"usage":{"output_tokens":1}}))
        w(sse("message_stop", {"type":"message_stop"}))

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
```

## Appendix B: `mcp_srv.py` (minimal stdio MCP server for testing)

Usage:
```bash
--mcp-config '{"mcpServers":{"mine":{"type":"stdio","command":"python3","args":["mcp_srv.py","2"]}}}'
```
The argument is the number of tools: 1 real `get_weather` tool plus N-1 filler tools.

```python
import json, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2

def tools():
    out = [{
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }]
    for i in range(1, N):
        out.append({"name": f"filler_{i}", "description": f"Filler tool number {i}. " + "Lorem ipsum. " * 20,
                    "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}})
    return out

for line in sys.stdin:
    msg = json.loads(line)
    mid, m = msg.get("id"), msg.get("method")
    if mid is None:
        continue
    if m == "initialize":
        res = {"protocolVersion": msg["params"].get("protocolVersion", "2025-06-18"),
               "capabilities": {"tools": {}}, "serverInfo": {"name": "mine", "version": "1"},
               "instructions": "These are my server instructions."}
    elif m == "tools/list":
        res = {"tools": tools()}
    elif m == "tools/call":
        res = {"content": [{"type": "text", "text": "sunny"}]}
    else:
        res = {}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}) + "\n")
    sys.stdout.flush()
```

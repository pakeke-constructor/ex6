# Agent Learnings: ctx.data, CodeEnv, and Runtime Tool Injection

## Problem Space

We needed runtime tool injection into code-mode (adding/removing tools mid-conversation).
This cascaded into deeper questions about data ownership, fork/clear semantics, and state persistence.

## Key Decisions & Rationale

### 1. Tool injection is a code-mode concept, not a Context concept

`ctx.inject_tool()` would be clean API but breaks provider caching for non-code-mode contexts (tool schemas are part of the cached API request structure). In code-mode, tool schemas never change — it's always just the single `run_tools` tool. So injection lives in `code_mode.py` via `inject_tool(ctx, fn)` / `remove_tool(ctx, fn)`.

### 2. ctx.data must be defensive (StrictDataDict)

`ctx.data` accepts only str/int/float/bool/None. This catches bugs at write-time rather than silently producing broken state on fork/clear/checkpoint. Previously, plugins stored complex objects (lists, dicts, closures) in `ctx.data` which caused subtle issues with `copy.copy()`, fork semantics, and JSON serialization.

CPython gotcha: `dict.__init__` and `dict.update` bypass `__setitem__` at C level. StrictDataDict overrides `__init__` (delegates to `update`) and `update` (iterates and calls `self[k] = v`) to ensure validation always runs.

### 3. ctx.data_volatile for complex/mutable objects

New field cleared on both fork and clear. Houses things like:
- `_code_env`: the CodeEnv instance
- `_codemode_base_tools`: dict of raw tool fns
- `_codemode_tools`: dict of injected tool fns

Injected tools live here — lost on fork. This is an acceptable tradeoff: the common case is base tools (in closure, always survive). Injected tools are session-scoped. If fork happens after injection, the forked context's messages may reference tools that aren't in the sandbox — but the error is loud (sandbox NameError), not silent corruption.

### 4. CodeEnv: globals dict with prepare/sync lifecycle

CodeEnv is a dict subclass used as the `globals` argument to `exec()`.

Critical discovery: CPython's `exec()` uses C-level `PyDict_SetItem`/`PyDict_GetItem` on the globals dict, completely bypassing Python-level `__setitem__`/`__getitem__` overrides. This means a proxy-style dict (intercept writes, redirect to ctx.data) is impossible for globals.

Solution: pre/post sync pattern.
- `prepare()`: called before each exec. Re-wraps all tools with fresh per-call tracking (results/threads/tool_infos lists). Loads persisted `cm:*` variables from ctx.data into the dict.
- `sync()`: called after each exec. Scans dict for new simple-type values, writes them back to ctx.data with `cm:` prefix. Functions, ToolResults, etc. are skipped (not simple types) and remain local to the CodeEnv (lost on fork — correct behavior).

This means LLM variables like `x = 42` persist across run_tools calls (via ctx.data round-trip) and survive fork (ctx.data is copied). Complex assignments like `x = read_file("a.py")` stay local and are lost — also correct, since ToolResults aren't meaningful across calls.

### 5. Namespace prefixing in ctx.data

All plugins use colon-namespaced keys in ctx.data:
- `cm:varname` — code-mode LLM variables
- `cp:index`, `cp:objective`, `cp:data` — checkpoint state (flattened from nested dict)
- `plan:id` — task planner focused plan
- `cc_session`, `cc_turn` — claude-code provider session state

Checkpoint's `cp:data` stores a JSON string of ctx.data at checkpoint time. On restore, `json.loads` + `StrictDataDict()` reconstructs it. This replaces the old `copy.copy(ctx.data)` approach which had issues with complex nested objects and recursive references.

### 6. System prompt stays static for cache hits

Injected tools do NOT mutate the code-mode system prompt. The caller is responsible for documenting new tools via user messages. The system prompt's `_snapshot` mechanism means once rendered, it's frozen — cache-friendly. Only the `run_tools` closure's sandbox env changes at runtime.

## Architecture Summary

```
ctx.data (StrictDataDict)          ctx.data_volatile (dict)
├── cm:x = 42                      ├── _code_env = CodeEnv(ctx)
├── cm:name = "foo"                ├── _codemode_base_tools = {name: fn}
├── cp:index = 5                   └── _codemode_tools = {name: fn}
├── cp:objective = "exploring"
├── cp:data = '{"cm:x": 42, ...}'
├── plan:id = "abc123"
└── cc_session = "uuid..."

CodeEnv (dict, used as exec globals)
├── __builtins__, _getattr_, modules  (static, set in __init__)
├── read_file, edit_file, ...         (re-wrapped per call in prepare())
├── cm:* vars loaded from ctx.data    (loaded in prepare())
└── LLM assignments during exec       (synced back in sync())
```

fork: copies ctx.data (cm:* vars survive), clears data_volatile (CodeEnv recreated lazily)
clear: keeps ctx.data, clears data_volatile
checkpoint/condense: snapshots/restores ctx.data as JSON string

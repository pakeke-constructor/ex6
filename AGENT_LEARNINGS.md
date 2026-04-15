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

### 4. CodeEnv: split globals/locals with proxy

CodeEnv is a dict subclass used as the `locals` argument to `exec()`. A separate plain `_globals` dict holds builtins, modules, and wrapped tools.

CPython's `exec()` bypasses Python-level `__setitem__`/`__getitem__` on the globals dict (C-level PyDict ops). But the locals argument can be any mapping — Python calls the mapping protocol properly.

So CodeEnv is passed as locals, where `__setitem__` proxies simple-type writes directly to `ctx.data` with `cm:` prefix, and `__getitem__` falls back to `ctx.data` for `cm:*` keys. No sync step needed.

- `prepare()`: called before each exec. Clears local state, re-wraps tools into `_globals` with fresh per-call tracking.
- Simple LLM variables (`x = 42`) are written to ctx.data immediately via `__setitem__`. They persist across calls and survive fork.
- Complex assignments (`x = read_file("a.py")`) stay only in the dict (not simple types), lost on fork — correct behavior.

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

CodeEnv._globals (plain dict, exec globals)
├── __builtins__, _getattr_, modules  (static, set in __init__)
└── read_file, edit_file, ...         (re-wrapped per call in prepare())

CodeEnv (dict, exec locals — mapping protocol)
├── __setitem__ proxies simple types to ctx.data with cm: prefix
├── __getitem__ falls back to ctx.data for cm:* keys
└── LLM assignments during exec       (simple types persist immediately, complex stay local)
```

fork: copies ctx.data (cm:* vars survive), clears data_volatile (CodeEnv recreated lazily)
clear: keeps ctx.data, clears data_volatile
checkpoint/condense: snapshots/restores ctx.data as JSON string

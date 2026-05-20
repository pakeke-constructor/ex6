
import typing

"""
code_mode: Python as a tool-calling interface for LLMs.

Instead of rigid JSON tool-call schemas, the LLM writes Python snippets to call
tools. This gives us composition, chaining, and parallelism for free — using
syntax the LLM already knows.

The LLM has a single tool, `run_tools`, whose `code` param is Python:

    run_tools(code='''
        read_file("main.py").print()
        read_file("utils.py").print()
        edit_file("config.py", old, new).status()
    ''')

## ToolResult

Every tool call returns a ToolResult (a future). The LLM controls what enters
its context window by choosing how to consume each result:

    .print()  non-blocking, show full result in context of caller agent. Use to READ data.
    .status() non-blocking, show OK or error to caller agent. Use to CONFIRM writes/actions.
    .get()    blocking, returns value silently. Use to PASS data to another tool.
    .is_ok()  blocking, returns bool. Use to BRANCH on success/failure.

If the LLM never calls .print()/.status(), the result is silently discarded.
This is by design; the LLM decides what's worth seeing.

.print() and .status() return self (the ToolResult), so they can be chained:

    x = read_file("main.py").print().get()   # print AND capture value
    edit_file("x.py", old, new).status().get()  # confirm AND get value

.get() raises on error — fail-fast so garbage data is never passed downstream.
.print() and .status() do NOT raise — they show the error in context instead.

## Examples

### Parallel reads (both run concurrently, both printed):

    read_file("main.py").print()
    read_file("utils.py").print()

### Write + confirm:

    edit_file("x.py", old, new).status()

### Subagents: the primary use case for chaining:

    # Explore codebase, then pass findings to a focused agent
    findings = explore_agent("find all database access patterns")
    review_agent("check for SQL injection risks", context=findings.get()).print()

    # Fan-out: two independent subagents, then combine
    a = explore_agent("find all API endpoints")
    b = explore_agent("find all auth middleware")
    review_agent("do all endpoints use auth?", apis=a.get(), auth=b.get()).print()

    # Sequential refinement
    draft = coding_agent("write a parser for this format", spec=read_file("spec.md").get())
    review = review_agent("find bugs", code=draft.get())
    if not review.is_ok():
        coding_agent("fix these issues", code=draft.get(), issues=review.get()).print()

### Branching on success:

    r = edit_file("config.json", old, new)
    if r.is_ok():
        restart_server().status()
    else:
        r.print()  # see the error

### Mixed: some results printed, some just status, some silent:

    read_file("main.py").print()           # need to see this
    write_file("out.py", code).status()    # just confirm it worked
    x = read_file("data.json")            # silent — only passing to next tool
    transform(x.get()).print()
"""

import inspect
import json
import math
import os.path
import re
import threading
import time
import types
import ex6



# Code-mode globals
PRELOADED_MODULES = {
    "re": re,
    "json": json,
    "math": math,
    "time": time,
    "os": types.SimpleNamespace(path=os.path),
}

_CM_PREFIX = "codemode:"

class CodeEnv(dict):
    """Locals mapping for code-mode exec(). Proxies simple-type vars to/from ctx.data."""

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self._globals = {
            "__builtins__": __builtins__,
            **PRELOADED_MODULES,
        }

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if not key.startswith("_") and isinstance(value, ex6.SIMPLE_DATA_TYPES):
            self.ctx.data[_CM_PREFIX + key] = value

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            cm_key = _CM_PREFIX + key
            if cm_key in self.ctx.data:
                return self.ctx.data[cm_key]
            raise

    def __contains__(self, key):
        return super().__contains__(key) or (_CM_PREFIX + key) in self.ctx.data

    def prepare(self, results, threads, tool_infos):
        """Re-wrap tools into _globals with fresh per-call tracking state."""
        self.clear()
        if isinstance(__builtins__, dict):
            builtins_dict = __builtins__.copy()
        else:
            builtins_dict = __builtins__.__dict__.copy()
        def _blocked_print(*args, **kwargs):
            raise RuntimeError("print() disabled in code-mode. Use ToolResult.print()")
        builtins_dict["print"] = _blocked_print
        self._globals["__builtins__"] = builtins_dict
        tools = dict(self.ctx.data_volatile.get('codemode:base_tools', {}))
        tools.update(self.ctx.data_volatile.get('codemode:tools', {}))
        for name, fn in tools.items():
            self._globals[name] = _wrap_tool_threaded(fn, self.ctx, results, threads, tool_infos)


class ToolResult:
    """Future-like object returned by tool calls."""
    __slots__ = ('value', '_error', '_event', '_call_str', '_fn_name', '_results')
    def __init__(self, call_str, results):
        self.value = None
        self._error = None
        self._event = threading.Event()
        self._call_str = call_str
        self._fn_name = call_str.split('(')[0]
        self._results = results
    def _set(self, val):
        self.value = val
        self._event.set()
    def _set_error(self, err):
        self._error = err
        self._event.set()
    def is_ok(self):
        self._event.wait()
        return self._error is None
    def get(self):
        self._event.wait()
        if self._error: raise self._error
        return self.value
    def print(self):
        self._results.append((self, "full"))
        return self
    def status(self):
        self._results.append((self, "status"))
        return self


def exec_sandboxed(code: str, env: CodeEnv):
    """Execute code in normal Python runtime."""
    compiled = compile(code, '<tools>', 'exec')
    exec(compiled, env._globals, env)



def _resolve_type_hint(h):
    if not isinstance(h, str):
        return h
    return {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
    }.get(h, h)



def _strong_isinstance(obj: object, type_hint: type) -> bool:
    """
    isinstance but with support for generics
    """
    type_hint = _resolve_type_hint(type_hint)
    if type_hint is typing.Any:
        return True
    if isinstance(type_hint, str):
        hint = type_hint.rsplit('.', 1)[-1]
        return type(obj).__name__ == hint

    origin = typing.get_origin(type_hint)
    args = typing.get_args(type_hint)

    # Fall back to regular isinstance for non-generic types
    if origin is None:
        return isinstance(obj, type_hint)

    # Optional[T] is Union[T, None], also handles X | Y | None unions
    if origin is typing.Union or origin is types.UnionType:
        return any(_strong_isinstance(obj, arg) for arg in args)

    # Check the outer container type first
    if not isinstance(obj, origin):
        return False

    # list[T]
    if origin is list:
        assert(isinstance(obj, list))
        (item_type,) = args
        return all(_strong_isinstance(item, item_type) for item in obj)

    return False


def _validate_tool_args(fn, args, kwargs):
    """Validate arg types against function annotations. Raises TypeError on mismatch."""
    sig = inspect.signature(fn)
    params = [(n, p) for n, p in sig.parameters.items() if n != 'ctx']
    # bind args to param names
    bound = {}
    for i, (name, p) in enumerate(params):
        if i < len(args):
            bound[name] = args[i]
        elif name in kwargs:
            bound[name] = kwargs[name]

    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}

    for name, val in bound.items():
        p = sig.parameters[name]
        ann = hints.get(name, p.annotation)
        if ann is inspect.Parameter.empty:
            continue
        ann = _resolve_type_hint(ann)
        if not _strong_isinstance(val, ann):
            ann_name = ann.__name__ if hasattr(ann, '__name__') else str(ann)
            raise TypeError(
                f"{fn.__name__}() param '{name}' expected {ann_name}, got {type(val).__name__}: {val!r}"
            )


def _wrap_tool_threaded(fn, ctx, results: list, threads: list, tool_infos: list):
    """Wrap tool to run in thread. Returns ToolResult with .get()/.print()/.status()."""
    def wrapper(*args, **kwargs):
        def _short(a, maxlen=40):
            s = repr(a)
            return s if len(s) <= maxlen else s[:maxlen] + '...'
        call_str = f'{fn.__name__}({", ".join([_short(a) for a in args] + [f"{k}={_short(v)}" for k, v in kwargs.items()])})'
        tr = ToolResult(call_str, results)
        try:
            _validate_tool_args(fn, args, kwargs)
        except TypeError as e:
            tr._set_error(e)
            tool_infos.append((fn.__name__, list(args), kwargs, None, tr))
            return tr
        def run():
            try: tr._set(fn(ctx, *args, **kwargs))
            except Exception as e:
                ex6.debug_print(f"tool {call_str} failed: {e}")
                tr._set_error(e)
        t = threading.Thread(target=run)
        t.start()
        threads.append(t)
        tool_infos.append((fn.__name__, list(args), kwargs, t, tr))
        return tr
    return wrapper



def _annotation_name(p):
    if p.annotation == inspect.Parameter.empty: return '?'
    return p.annotation.__name__ if hasattr(p.annotation, '__name__') else str(p.annotation)

def generate_tool_desc(fn) -> str:
    """Generate a description string for a tool function: name, args w/ types, and docstring."""
    sig = inspect.signature(fn)
    params = [(n, p) for n, p in sig.parameters.items() if n != 'ctx']
    args = ", ".join(
        f"{n}: {_annotation_name(p)}"
        + (f" = {p.default!r}" if p.default != inspect.Parameter.empty else "")
        for n, p in params
    )
    doc = (fn.__doc__ or "").strip()
    sig_line = f"{fn.__name__}({args})"
    body = f"{sig_line}\n{doc}" if doc else sig_line
    return f"<tool {fn.__name__}>\n{body}\n</tool>"


def _get_code_env(ctx, tools):
    """Get or create the CodeEnv for this context. Stores base tools on first call."""
    env = ctx.data_volatile.get('codemode:env')
    if not env:
        ctx.data_volatile['codemode:base_tools'] = {fn.__name__: fn for fn in tools}
        env = CodeEnv(ctx)
        ctx.data_volatile['codemode:env'] = env
    return env


def make_code_mode_tool(tools: list):
    """Create the run_tools tool function for Python code execution."""
    for fn in tools:
        ex6._validate_tool_sig(fn.__name__, fn)
    def run_tools(ctx: ex6.Context, code="", tool_call_id=None):
        """Execute tool calls as Python code.
        - Do NOT use import statements.
        - Tools return a ToolResult. You MUST call .print() or .status() to see results, or call .get() or .is_ok() to use the result for another call."""
        results, threads, tool_infos = [], [], []
        env = _get_code_env(ctx, tools)
        env.prepare(results, threads, tool_infos)

        exec_error = None
        if tool_call_id:
            def code_render(buf, x, y, w):
                row = 0
                if exec_error:
                    ex6.render_tool_line(buf, x, y + row, w, "run_tools", [], 'error', str(exec_error))
                    row += 1
                for name, args, kwargs, t, tr in tool_infos:
                    alive = t is not None and t.is_alive()
                    status = 'running' if alive else ('error' if tr._error else 'ok')
                    detail = None if alive else (str(tr._error) if tr._error else None)
                    ex6.render_tool_line(buf, x, y + row, w, name, args, status, detail, kwargs)
                    row += 1
                return max(row, 1)
            ctx.set_tool_renderer(tool_call_id, code_render)
        try:
            exec_sandboxed(code, env)
        except Exception as e:
            exec_error = e
            for t in threads: t.join()
            raise ValueError(f"exec failed: {e}")
        for t in threads: t.join()
        if results:
            full_results = [(tr, mode) for tr, mode in results if mode == "full"]
            fn_names = [tr._fn_name for tr, mode in full_results]
            duplicate_names = {n for n in fn_names if fn_names.count(n) > 1}
            parts = []
            for tr, mode in results:
                if mode == "status":
                    val = "OK" if tr._error is None else str(tr._error)
                    parts.append(f"<tool_status {tr._call_str}>{val}</tool_status>")
                else:
                    val = str(tr._error) if tr._error else tr.value
                    if len(full_results) == 1:
                        # only 1 result -> just append the result string directly
                        parts.append(val)
                    elif tr._fn_name in duplicate_names:
                        # more than 1 result, with duplicate type -> append full call-sig
                        parts.append(f"<tool_result {tr._call_str}>\n{val}\n</tool_result>")
                    else:
                        # more than 1 result, but all different types -> append func-names only.
                        parts.append(f"<tool_result {tr._fn_name}>\n{val}\n</tool_result>")
            return "\n\n".join(parts)
        return "No output. (Use `.print()` or `.status()` if you want to see results)"
    return run_tools


def inject_tool(ctx, fn):
    """Add a tool to this context's code-mode sandbox. Available next run_tools call."""
    ex6._validate_tool_sig(fn.__name__, fn)
    tools = ctx.data_volatile.setdefault('codemode:tools', {})
    tools[fn.__name__] = fn

def remove_tool(ctx, fn):
    """Remove an injected tool from this context's code-mode sandbox."""
    tools = ctx.data_volatile.get('codemode:tools', {})
    tools.pop(fn.__name__, None)


RUN_TOOLS_NAME = "run_tools"

COMMON_MISTAKES = """
<common_mistakes>
COMMON MISTAKES — do NOT do these:
- Tool calls do nothing unless you consume ToolResult with `.print()`, `.status()`, `.get()`, or `.is_ok()`.

run_tools```
# BAD — since there's no `.print()` or `.status()`, result is silently discarded, you will see NOTHING:
read_file("a.py")

# BAD — print doesn't exist, this WILL error.
print(read_file("a.py"))
```

run_tools```
# GOOD — .print() injects result into your context:
read_file("a.py").print()

# GOOD — .status() confirms success:
edit_file("a.py", old, new).status()

# GOOD — .get() passes data to another tool:
data = read_file("a.py").get()
search(data).print()

# GOOD — silent chaining via .get() (intermediate result not printed):
search(read_file("a.py").get()).print()
</common_mistakes>
```
"""

def make_code_mode_system_prompt(tools: list, include_common_mistakes: bool = True) -> ex6.Message:
    """System prompt + run_tools tool for Python code execution."""
    sorted_tools = sorted(tools, key=lambda f: f.__name__)
    tool_docs = ("\n").join(generate_tool_desc(fn) for fn in sorted_tools)
    run_tools = make_code_mode_tool(tools)
    common_mistakes = (include_common_mistakes and COMMON_MISTAKES) or ""
    return ex6.Message(role="system", overview="tools", content=f"""\
<tools>
Use the `{RUN_TOOLS_NAME}` tool to invoke tools. This is your interface for file/system actions.
The `code` param is a normal Python environment, however you are given tools that you should prefer to use.
(IMPORTANT: `print()` is disabled in code-mode and raises error; use `ToolResult.print()` instead.)
Combine multiple calls in a single run_tools block — they execute in parallel.

These modules are pre-loaded for convenience:
- `re` — regex: re.search(), re.findall(), re.sub(), etc.
- `json` — json.loads(), json.dumps()
- `math` — math.ceil(), math.floor(), math.log(), etc.
- `time` — time.sleep(), time.time()
- `os.path` — os.path.join(), os.path.basename(), os.path.dirname(), etc.

<tool_results>
Every tool call returns a ToolResult, which is a future containing the task's output and status.
On their own, tool-calls don't output anything in your context window.
You MUST call one of these to see output:
- `ToolResult.print()` — non-blocking. injects the FULL result into your context. Returns self (ToolResult object)
- `ToolResult.status()` — non-blocking. injects OK or ERROR into your context. Use for writes/actions you don't need to read. Returns self (ToolResult object)
- `ToolResult.get()` — blocking. returns the value silently. Use to pass data to another tool.
- `ToolResult.is_ok()` — blocking. returns the value silently. Use to branch depending on whether another tool succeeded.

IMPORTANT: If you do not call .print() or .status(), you will NOT see the result AT ALL.
</tool_results>

<available_tools>
{tool_docs}
</available_tools>

<tool_examples>
{RUN_TOOLS_NAME}```
# Read files — .print() to see contents
read_file("main.py").print()
read_file("utils.py").print()
```

{RUN_TOOLS_NAME}```
# Write file — .status() to confirm success
edit_file("src/main.lua",
r'''function Player:update(dt)
    self.x = self.x + 1
end''',
r'''function Player:update(dt)
    self.x = self.x + self.speed * dt
    self.y = self.y + self.vy * dt
end'''
).status()
```

{RUN_TOOLS_NAME}```
# Chain: pass data from one tool to another
x = read_file("schema.sql") # `x` is a ToolResult
x.print()
search(x.get()).print()

# Silent chaining: pass result without printing intermediate ToolResult
search(read_file("schema.sql").get()).print()
```
</tool_examples>
{common_mistakes}
</tools>
""", tools=[run_tools])


# ==================== RENDERERS ====================

# extract_tags: scan lines for <tag header>content</tag>, remove them, return [(tag, header, content)]
# works for single-line (<tag h>c</tag>) and multi-line (<tag h>\n...\n</tag>)
def extract_tags(lines, *tags):
    found = []
    i = 0
    while i < len(lines):
        if not isinstance(lines[i], str): i += 1; continue
        tag = next((t for t in tags if lines[i].startswith(f'<{t} ')), None)
        if not tag: i += 1; continue
        close = f'</{tag}>'
        if close in lines[i]:  # single-line
            gt = lines[i].index('>')
            found.append((tag, lines[i][len(tag)+2:gt], lines[i][gt+1:].removesuffix(close)))
            del lines[i]
        else:  # multi-line
            header = lines[i][len(tag)+2:].rstrip('>')
            j, parts = i + 1, []
            while j < len(lines):
                if isinstance(lines[j], str) and lines[j].strip() == close: break
                if isinstance(lines[j], str): parts.append(lines[j])
                j += 1
            found.append((tag, header, '\n'.join(parts)))
            del lines[i:j+1]
    return found



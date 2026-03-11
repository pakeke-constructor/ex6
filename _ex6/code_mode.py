"""
code_mode: sandboxed Python as a tool-calling interface for LLMs.

Instead of rigid JSON tool-call schemas, the LLM writes Python snippets to call
tools. This gives us composition, chaining, and parallelism for free — using
syntax the LLM already knows.

The LLM has a single tool, `run_tools`, whose `code` param is sandboxed Python:

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
import threading
import time
import ex6
from RestrictedPython import compile_restricted_exec


# Sandbox setup
SAFE_BUILTINS = {
    "None": None, "True": True, "False": False,
    "int": int, "float": float, "bool": bool, "complex": complex,
    "abs": abs, "round": round, "pow": pow,
    "list": list, "tuple": tuple, "set": set, "dict": dict, "frozenset": frozenset,
    "range": range, "len": len, "enumerate": enumerate, "zip": zip,
    "min": min, "max": max, "sum": sum, "all": all, "any": any,
    "str": str, "repr": repr, "format": format,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
}

class ToolResult:
    """Future-like object returned by tool calls."""
    __slots__ = ('value', '_error', '_event', '_call_str', '_results')
    def __init__(self, call_str, results):
        self.value = None
        self._error = None
        self._event = threading.Event()
        self._call_str = call_str
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


def _no_import(*args, **kwargs):
    raise ImportError("imports disabled")


def exec_sandboxed(code: str, env: dict):
    """Execute code in RestrictedPython sandbox."""
    sandbox_globals = {"__builtins__": SAFE_BUILTINS.copy()}
    sandbox_globals["__import__"] = _no_import
    def _getattr_(obj, name):
        if isinstance(obj, ToolResult) and name in ("get", "print", "status", "is_ok"):
            return getattr(obj, name)
        raise AttributeError(f"no attribute {name}")
    sandbox_globals["_getattr_"] = _getattr_
    sandbox_globals.update(env)

    result = compile_restricted_exec(code, '<tools>')
    if result.errors:
        raise SyntaxError(f"restricted compile: {result.errors}")
    exec(result.code, sandbox_globals)


def _wrap_tool_threaded(fn, ctx, results: list, threads: list, tool_infos: list):
    """Wrap tool to run in thread. Returns ToolResult with .get()/.print()/.status()."""
    def wrapper(*args, **kwargs):
        def _short(a, maxlen=40):
            s = repr(a)
            return s if len(s) <= maxlen else s[:maxlen] + '...'
        call_str = f'{fn.__name__}({", ".join(_short(a) for a in args)})'
        tr = ToolResult(call_str, results)
        def run():
            try: tr._set(fn(ctx, *args, **kwargs))
            except Exception as e:
                ex6.debug_print(f"tool {call_str} failed: {e}")
                tr._set_error(e)
        t = threading.Thread(target=run)
        t.start()
        threads.append(t)
        tool_infos.append((call_str, t, tr))
        return tr
    return wrapper



def generate_tool_desc(fn) -> str:
    """Generate a description string for a tool function: name, args w/ types, and docstring."""
    sig = inspect.signature(fn)
    params = [(n, p) for n, p in sig.parameters.items() if n != 'ctx']
    args = ", ".join(
        f"{n}: {p.annotation.__name__ if p.annotation != inspect.Parameter.empty else '?'}"
        + (f" = {p.default!r}" if p.default != inspect.Parameter.empty else "")
        for n, p in params
    )
    doc = (fn.__doc__ or "").strip()
    sig_line = f"{fn.__name__}({args})"
    body = f"{sig_line}\n{doc}" if doc else sig_line
    return f"<tool {fn.__name__}>\n{body}\n</tool>"


def make_code_mode_tool(tools: list):
    """Create the run_tools tool function for sandboxed code execution."""
    def run_tools(ctx, code="", tool_call_id=None):
        """Execute tool calls as Python code.
        - Do NOT use import statements.
        - Tools return a ToolResult. You MUST call .print() or .status() to see results."""
        results, threads, tool_infos = [], [], []
        env = {fn.__name__: _wrap_tool_threaded(fn, ctx, results, threads, tool_infos) for fn in tools}

        if tool_call_id:
            def code_render(buf, x, y, w):
                row = 0
                for call_str, t, tr in tool_infos:
                    alive = t.is_alive()
                    status = 'running' if alive else ('error' if tr._error else 'ok')
                    detail = None if alive else (str(tr._error) if tr._error else None)
                    _tool_line(buf, x, y + row, w, call_str, status, detail)
                    row += 1
                return max(row, 1)
            ctx.set_tool_renderer(tool_call_id, code_render)
        try:
            exec_sandboxed(code, env)
        except Exception as e:
            for t in threads: t.join()
            raise ValueError(f"exec failed: {e}")
        for t in threads: t.join()
        if results:
            parts = []
            for tr, mode in results:
                if mode == "status":
                    val = "OK" if tr._error is None else str(tr._error)
                    parts.append(f"<tool_status {tr._call_str}>{val}</tool_status>")
                else:
                    val = str(tr._error) if tr._error else tr.value
                    parts.append(f"<tool_result {tr._call_str}>\n{val}\n</tool_result>")
            return "\n\n".join(parts)
        return "No output. (Use `.print()` or `.status()` if you want to see results)"
    return run_tools


def make_code_mode_system_prompt(tools: list) -> ex6.Message:
    """System prompt + run_tools tool for sandboxed code execution."""
    names = ", ".join(fn.__name__ for fn in tools)
    run_tools = make_code_mode_tool(tools)
    return ex6.Message(role="system", content=f"""\
# Tools
Use the `run_tools` tool. The `code` param is sandboxed Python.
IMPORTANT: imports are NOT available. Do NOT use `import`, `from X import`, or `__import__`. Only the listed functions exist.
Combine multiple calls in a single run_tools block — they execute in parallel.
Available: {names}""", tools={"run_tools": run_tools})


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

SPINNER = ['/', '-', '\\', '|']

def _tool_line(buf, x, y, w, label, status='ok', detail=None):
    icon, color = {'running': (SPINNER[int(time.time()*8)%4], 'yellow'), 'error': ('x', 'red')}.get(status, ('v', 'green'))
    buf.puts(x, y, f"[{icon}]", txt_color=color, style='bold')
    buf.puts(x+4, y, label[:w-4], txt_color='blue')
    if detail:
        buf.puts(x+5+len(label), y, detail.replace('\n',' ')[:w-6-len(label)], txt_color='bright_black')



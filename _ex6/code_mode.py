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
    __slots__ = ('value', '_event', '_call_str', '_results')
    def __init__(self, call_str, results):
        self.value = None
        self._event = threading.Event()
        self._call_str = call_str
        self._results = results
    def _set(self, val):
        self.value = val
        self._event.set()
    def get(self):
        self._event.wait()
        if isinstance(self.value, str) and self.value.startswith("ERROR:"):
            raise ValueError(self.value)
        return self.value
    def print(self):
        self._event.wait()
        self._results.append({"call": self._call_str, "value": self.value, "mode": "full"})
        return self.value
    def is_ok(self):
        self._event.wait()
        return not (isinstance(self.value, str) and self.value.startswith("ERROR:"))
    def status(self):
        self._event.wait()
        ok = self.is_ok()
        self._results.append({"call": self._call_str, "value": "OK" if ok else self.value, "mode": "status"})
        return self.value


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


def _wrap_tool_threaded(fn, ctx, results: list, threads: list):
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
                tr._set(f"ERROR: {e}")
        t = threading.Thread(target=run)
        t.start()
        threads.append(t)
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
    return f"- {fn.__name__}({args})\n  {doc}" if doc else f"- {fn.__name__}({args})"


def make_code_mode_tool(tools: list):
    """Create the run_tools tool function for sandboxed code execution."""
    def run_tools(ctx, code=""):
        """Execute tool calls as Python code.
        - Do NOT use import statements.
        - Tools return a ToolResult. You MUST call .print() or .status() to see results."""
        results, threads = [], []
        env = {fn.__name__: _wrap_tool_threaded(fn, ctx, results, threads) for fn in tools}
        try:
            exec_sandboxed(code, env)
        except Exception as e:
            for t in threads: t.join()
            raise ValueError(f"exec failed: {e}")
        for t in threads: t.join()
        if results:
            parts = []
            for r in results:
                if r.get("mode") == "status":
                    parts.append(f"<tool_status {r['call']}>{r['value']}</tool_status>")
                else:
                    parts.append(f"<tool_result {r['call']}>\n{r['value']}\n</tool_result>")
            return "\n\n".join(parts)
        return "No tools were called."
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

SPINNER = ['/', '-', '\\', '|']

def make_tools_renderer(code: str, ctx: ex6.Context) -> ex6.RenderFn:
    lines = [ln.strip() for ln in code.strip().split('\n') if ln.strip()]
    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        frame = int(time.time() * 8) % 4
        running = ctx.llm_suspended
        icon = SPINNER[frame] if running else '...'
        for i, call in enumerate(lines):
            buf.puts(x, y + i, f"[{icon}]", txt_color='yellow', style='bold')
            buf.puts(x + 6, y + i, call[:w-4], txt_color='blue')
        return len(lines)
    return render


@ex6.output_renderer
def render_tools_block(role: str, output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    i = 0
    while i < len(output):
        line = output[i]
        if isinstance(line, str) and '"run_tools"' in line:
            try:
                tc = json.loads(line.strip())
                if tc.get("name") == "run_tools":
                    code = tc["args"]["code"]
                    output[i] = make_tools_renderer(code, ctx)
                    i += 1
                    continue
            except: pass
        i += 1


def make_tool_result_renderer(call: str, content: str) -> ex6.RenderFn:
    is_error = content.startswith("ERROR:")
    preview = content[:50].replace('\n', ' ')
    def render(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
        if is_error:
            buf.puts(x, y, "[x]", txt_color='red', style='bold')
        else:
            buf.puts(x, y, "[v]", txt_color='green', style='bold')
        buf.puts(x + 4, y, call, txt_color='blue')
        buf.puts(x + 5 + len(call), y, preview[:w - 6 - len(call)], txt_color='bright_black')
        return 1
    return render


@ex6.output_renderer
def render_tool_results(role: str, output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    i = 0
    while i < len(output):
        line = output[i]
        if isinstance(line, str) and line.startswith('<tool_result '):
            call = line[13:].rstrip('>')
            j = i + 1
            content_lines = []
            while j < len(output):
                if isinstance(output[j], str) and output[j].strip() == '</tool_result>':
                    break
                if isinstance(output[j], str):
                    content_lines.append(output[j])
                j += 1
            content = '\n'.join(content_lines)
            del output[i:j+1]
            output.insert(i, make_tool_result_renderer(call, content))
        i += 1

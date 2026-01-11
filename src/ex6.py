
from typing import Optional

import threading
import time
import glob
import os
import inspect
from dataclasses import dataclass, field
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
import readchar



_commands = {}

_tools = {}



class LockedValue:
    def __init__(self, val):
        self._val = val
        self._lock = threading.Lock()
    def take(self):
        with self._lock:
            v = self._val
            self._val = []
            return v
    def append(self, x):
        with self._lock:
            self._val.append(x)


@dataclass
class ContextInfo:
    name: str
    model: str = "opus-4.5"
    tokens: int = 32000
    max_tokens: int = 200000
    cost: float = 0.15
    messages: list = field(default_factory=list)
    children: list = field(default_factory=list)

# Dummy data
DUMMY_CONTEXTS = [
    ContextInfo("ctx1", messages=[("sys-prompt-1", 12000), ("user", 400)]),
    ContextInfo("ctx2", children=[
        ContextInfo("ctx2_child", tokens=8000),
        ContextInfo("blah_second_child", children=[
            ContextInfo("nested_child", tokens=2000)
        ])
    ]),
    ContextInfo("foobar", model="sonnet-4", tokens=45000, cost=0.08),
    ContextInfo("debug_ctx", tokens=5000, messages=[("sys", 4000), ("user", 1000)]),
]

def flatten_contexts(ctxs, depth=0):
    """Flatten context tree into (ctx, depth) pairs."""
    result = []
    for c in ctxs:
        result.append((c, depth))
        result.extend(flatten_contexts(c.children, depth + 1))
    return result

@dataclass
class AppState:
    input_buffer: str = ""
    console: list = field(default_factory=list)
    keys: LockedValue = field(default_factory=lambda: LockedValue([]))
    input_stack: list = field(default_factory=list)
    current_context: Optional['ContextInfo'] = None
    mode: str = "selection"
    hover_idx: int = 0
    contexts: list = field(default_factory=lambda: DUMMY_CONTEXTS)

state = AppState()



def get_fn_name(fn):
    return fn.__name__


def command(fn):
    '''
    used like:

    @ex6.command
    def my_command(arg1, arg2): pass

    now, `/command a b` should be valid command
    '''
    name = get_fn_name(fn)
    sig = inspect.signature(fn)
    spec = [(p.name, p.annotation if p.annotation != inspect.Parameter.empty else str)
            for p in sig.parameters.values()]
    _commands[name] = (fn, spec)
    return fn


def tool(fn):
    '''
    @ex6.tool
    def my_llm_tool(arg1, arg2):
        pass

    can be included in ctx windows for LLMs.
    '''
    name = get_fn_name(fn)
    sig = inspect.signature(fn)
    spec = [(p.name, p.annotation if p.annotation != inspect.Parameter.empty else str)
            for p in sig.parameters.values()]
    _tools[name] = (fn, spec)
    return fn


class InputPass:
    '''Created every frame with keys pressed that frame.'''
    def __init__(self, keys: list):
        self._keys = set(keys)
        self._consumed = set()

    def consume(self, key: str) -> bool:
        if key in self._keys and key not in self._consumed:
            self._consumed.add(key)
            return True
        return False

    def consume_text(self) -> str:
        text = ""
        for k in list(self._keys - self._consumed):
            if len(k) == 1 and k.isprintable():
                self._consumed.add(k)
                text += k
        return text

    def consume_enter(self) -> bool:
        return self.consume(readchar.key.ENTER)
    def consume_backspace(self) -> bool:
        return self.consume(readchar.key.BACKSPACE)
    def consume_left(self) -> bool:
        return self.consume(readchar.key.LEFT)
    def consume_right(self) -> bool:
        return self.consume(readchar.key.RIGHT)
    def consume_up(self) -> bool:
        return self.consume(readchar.key.UP)
    def consume_down(self) -> bool:
        return self.consume(readchar.key.DOWN)


def dispatch_command(text):
    if not text.startswith("/"): return False
    parts = text[1:].split()
    if not parts: return False
    
    name, args = parts[0], parts[1:]
    if name not in _commands: return True
    
    fn, spec = _commands[name]
    return fn(*[typ(args[i]) for i, (_, typ) in enumerate(spec)])



def load_plugins():
    plugin_dir = os.path.join(os.path.dirname(__file__) or ".", ".ex6")
    if not os.path.isdir(plugin_dir):
        return
    for path in glob.glob(os.path.join(plugin_dir, "*.py")):
        with open(path, "r", encoding="utf-8") as f:
            exec(compile(f.read(), path, "exec"), {"__name__": "__plugin__", "__file__": path})



def input_thread():
    while True:
        key = readchar.readkey()
        state.keys.append(key)

def push_ui(draw_fn):
    state.input_stack.append(draw_fn)



def render_left_panel(inpt):
    flat = flatten_contexts(state.contexts)

    if inpt.consume_up():
        state.hover_idx = max(0, state.hover_idx - 1)
    if inpt.consume_down():
        state.hover_idx = min(len(flat) - 1, state.hover_idx + 1)
    if inpt.consume_enter() and flat:
        state.current_context, _ = flat[state.hover_idx]
        state.mode = "work"

    lines = Text()
    for i, (ctx, depth) in enumerate(flat):
        indent = "    " * depth
        prefix = ">> " if i == state.hover_idx else "   "
        style = "bold cyan" if i == state.hover_idx else ""
        lines.append(f"{prefix}{indent}{ctx.name}\n", style=style)
    return Panel(lines, title="Contexts")


def render_right_panel():
    flat = flatten_contexts(state.contexts)
    if not flat:
        return Panel("No contexts", title="Info")
    ctx, _ = flat[state.hover_idx]

    ratio = ctx.tokens / ctx.max_tokens
    bar_len = 20
    filled = int(ratio * bar_len)
    bar = "[" + "X" * filled + "-" * (bar_len - filled) + "]"

    info = Text()
    info.append(f"{ctx.name}    ", style="bold")
    info.append(f"{ctx.model}\n", style="dim")
    info.append(f"{bar}\n")
    info.append(f"{ctx.tokens//1000}k / {ctx.max_tokens//1000}k tokens, ${ctx.cost:.2f}\n")
    info.append("─" * 20 + "\n")

    if ctx.messages:
        for name, toks in ctx.messages:
            info.append(f"{name} ({toks//1000}k)\n")
    else:
        info.append("(no messages)\n", style="dim")

    return Panel(info, title="Info")


def _render(inpt):
    layout = Layout()
    layout.split_row(
        Layout(render_left_panel(inpt), name="left"),
        Layout(render_right_panel(), name="right"),
    )
    return layout




if __name__ == "__main__":
    load_plugins()
    threading.Thread(target=input_thread, daemon=True).start()

    with Live(Layout(), screen=True, auto_refresh=False) as live:
        while True:
            inpt = InputPass(state.keys.take())
            if inpt.consume(readchar.key.CTRL_C):
                break
            live.update(_render(inpt), refresh=True)
            time.sleep(1/60)




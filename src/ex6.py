
from typing import Optional, Callable

import threading
import time
import copy
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



def mock_llm_stream():
    """Mock LLM data stream."""
    for _ in range(60):
        time.sleep(0.1)
        yield "token "



all_contexts = set()

def is_prefix(short_msgs, long_msgs):
    """Check if short_msgs is a prefix of long_msgs. Empty lists are not valid prefixes."""
    if not short_msgs or len(short_msgs) >= len(long_msgs):
        return False
    return long_msgs[:len(short_msgs)] == short_msgs

def find_parent(ctx):
    """Find the context with longest message prefix (direct parent)."""
    best = None
    for candidate in all_contexts:
        if candidate is ctx:
            continue
        if is_prefix(candidate.messages, ctx.messages):
            if best is None or len(candidate.messages) > len(best.messages):
                best = candidate
    return best

def get_children(ctx):
    """Get all direct children of this context."""
    return [c for c in all_contexts if find_parent(c) is ctx]

@dataclass
class ContextInfo:
    name: str
    model: str = "opus-4.5"
    llm_current_output: str = ""
    llm_currently_running: bool = False
    tokens: int = 32000
    max_tokens: int = 200000
    cost: float = 0.15
    messages: list = field(default_factory=list)
    input_stack: list = field(default_factory=list)

    def __post_init__(self):
        input_text = ""

        def console_input(inpt):
            nonlocal input_text
            input_text += inpt.consume_text()
            if inpt.consume_backspace() and input_text:
                input_text = input_text[:-1]
            if inpt.consume_enter() and input_text:
                text = input_text
                input_text = ""
                if text.startswith("/"):
                    dispatch_command(text)
                else:
                    self.call(text)
            return Panel(f"> {input_text}_", style="dim")

        # can add more 
        self.input_stack = [console_input]
        all_contexts.add(self)
    
    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other
    
    def call(self, text):
        cpy = self.fork()
        cpy.messages.append({"role": "user", "content": text})
        cpy.llm_currently_running = True
        cpy.llm_current_output = ""
        state.current_context = cpy  # switch view to new context

        def run():
            for token in mock_llm_stream():
                cpy.llm_current_output += token
            cpy.messages.append({"role": "assistant", "content": cpy.llm_current_output})
            cpy.llm_currently_running = False

        threading.Thread(target=run, daemon=True).start()

    def fork(self) -> ContextInfo:
        cpy = copy.copy(self)
        cpy.messages = copy.deepcopy(self.messages)
        cpy.input_stack = []
        cpy.__post_init__()  # fresh input handlers
        all_contexts.add(cpy)
        return cpy

    def push_ui(self, draw_fn):
        self.input_stack.append(draw_fn)


def get_content(msg: dict[str, str|Callable[[ContextInfo],str]], ctx: ContextInfo) -> str:
    c = msg["content"]
    return c(ctx) if callable(c) else c


# Parent-child relationships are auto-detected via message prefixes
_base_msg = {"role": "system", "content": "base"}
DUMMY_CONTEXTS = [
    ContextInfo("ctx1", messages=[
        {"role": "system", "content": "You are helpful.", "name": "sys-prompt-1"},
        {"role": "user", "content": "hello"},
    ]),
    ContextInfo("ctx2", messages=[_base_msg]),
    ContextInfo("ctx2_child", tokens=8000, messages=[_base_msg, {"role": "user", "content": "child1"}]),
    ContextInfo("blah_second_child", messages=[_base_msg, {"role": "user", "content": "child2"}]),
    ContextInfo("nested_child", tokens=2000, messages=[_base_msg, {"role": "user", "content": "child2"}, {"role": "assistant", "content": "nested"}]),
    ContextInfo("foobar", model="sonnet-4", messages=[_base_msg], tokens=45000, cost=0.08),
    ContextInfo("debug_ctx", tokens=5000, messages=[
        {"role": "system", "content": "Debug mode."},
        {"role": "user", "content": "test input"},
    ]),
]


def flatten_contexts(ctxs, pinned=None):
    """Flatten context tree, collapsing single-child chains. Pinned ctx never collapsed."""
    def subtree(ctx, depth):
        children = get_children(ctx)
        if len(children) == 1 and ctx is not pinned:
            return subtree(children[0], depth)
        result = [(ctx, depth)]
        for child in children:
            result.extend(subtree(child, depth + 1))
        return result

    result = []
    roots = [c for c in ctxs if find_parent(c) is None]
    for c in roots:
        result.extend(subtree(c, 0))
    return result

def make_selection_input():
    input_text = ""

    def draw(inpt):
        nonlocal input_text
        input_text += inpt.consume_text()
        if inpt.consume_backspace() and input_text:
            input_text = input_text[:-1]
        if inpt.consume_enter() and input_text:
            text = input_text
            input_text = ""
            if text.startswith("/"):
                dispatch_command(text)
        return Panel(f"> {input_text}_", style="dim")

    return draw

@dataclass
class AppState:
    console: list = field(default_factory=list)
    keys: LockedValue = field(default_factory=lambda: LockedValue([]))
    current_context: Optional['ContextInfo'] = None
    mode: str = "selection"
    contexts: list = field(default_factory=lambda: DUMMY_CONTEXTS)
    selection_input: callable = field(default_factory=make_selection_input)

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



def dispatch_command(text: str):
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



ESCAPE = '\x1b'

SPINNER = "/-\\|/-\\||"


def render_left_panel(inpt):
    # LEFT navigates to parent
    if inpt.consume_left() and state.current_context:
        state.current_context = find_parent(state.current_context) or state.current_context

    flat = flatten_contexts(state.contexts, pinned=state.current_context)

    idx = next((i for i, (c, _) in enumerate(flat) if c is state.current_context), 0)

    if inpt.consume_up() and idx > 0:
        state.current_context = flat[idx - 1][0]
        idx -= 1
    if inpt.consume_down() and idx < len(flat) - 1:
        state.current_context = flat[idx + 1][0]
        idx += 1
    if inpt.consume_enter() and flat:
        state.mode = "work"

    spin_char = SPINNER[int(time.time() * 10) % len(SPINNER)]
    lines = Text()
    for i, (ctx, depth) in enumerate(flat):
        indent = "  " * depth
        prefix = "> " if i == idx else "  "
        style = "bold cyan" if i == idx else ""
        spin = f" {spin_char}" if ctx.llm_currently_running else ""
        lines.append(f"{prefix}{indent}{ctx.name}{spin}\n", style=style)
    return Panel(lines, title="Contexts")


def render_right_panel():
    ctx = state.current_context
    if not ctx:
        return Panel("No contexts", title="Info")

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
        for msg in ctx.messages:
            name = msg.get("name") or msg["role"]
            toks = len(get_content(msg, ctx)) * 4
            info.append(f"{name} ({toks//1000}k)\n")
    else:
        info.append("(no messages)\n", style="dim")

    return Panel(info, title="Info")


def render_input_box(inpt):
    ctx = state.current_context
    if ctx and ctx.input_stack:
        result = ctx.input_stack[-1](inpt)
        if result is None:  # signals pop
            ctx.input_stack.pop()
            return render_input_box(inpt)  # render next in stack
        return result
    return state.selection_input(inpt)  # selection-mode fallback


def render_selection_mode(inpt: InputPass):
    main = Layout()
    main.split_row(
        Layout(render_left_panel(inpt), name="left"),
        Layout(render_right_panel(), name="right"),
    )
    layout = Layout()
    layout.split_column(
        Layout(main, name="main"),
        Layout(render_input_box(inpt), name="input", size=3),
    )
    return layout


def render_work_mode(inpt: InputPass) -> Layout:
    ctx = state.current_context

    # ESC to go back to selection mode
    if inpt.consume(ESCAPE):
        state.mode = "selection"

    # Build conversation display
    conv = Text()
    if ctx and ctx.messages:
        for msg in ctx.messages:
            name = msg.get("name") or msg["role"]
            content = get_content(msg, ctx)
            conv.append(f"[{name}] ", style="bold cyan")
            conv.append(f"{content}\n", style="dim")
    else:
        conv.append("(empty conversation)\n", style="dim")

    # Show streaming output
    if ctx and ctx.llm_currently_running:
        conv.append(f"[assistant] ", style="bold yellow")
        conv.append(f"{ctx.llm_current_output}_\n", style="yellow")

    # Layout: conversation panel + input box
    layout = Layout()
    layout.split_column(
        Layout(Panel(conv, title=ctx.name if ctx else "Work"), name="main"),
        Layout(render_input_box(inpt), name="input", size=3),
    )
    return layout


def _render(inpt: InputPass):
    if state.mode == "selection":
        return render_selection_mode(inpt)
    else:
        return render_work_mode(inpt)



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




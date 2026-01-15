
from typing import Optional, Callable

import math
import threading
import time
import copy
import glob
import os
import inspect
from dataclasses import dataclass, field
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Static



_commands = {}
_tools = {}



def mock_llm_stream():
    """Mock LLM data stream."""
    for _ in range(60):
        time.sleep(0.1)
        yield "token "




def make_input(on_submit):
    text = ""
    cursor = 0

    def draw(inpt):
        nonlocal text, cursor
        # typing
        typed = inpt.consume_text()
        if typed:
            text = text[:cursor] + typed + text[cursor:]
            cursor += len(typed)
        # navigation
        if inpt.consume("left") and cursor > 0:
            cursor -= 1
        if inpt.consume_right() and cursor < len(text):
            cursor += 1
        # deletion
        if inpt.consume_backspace() and cursor > 0:
            text = text[:cursor-1] + text[cursor:]
            cursor -= 1
        if inpt.consume('\x7f') and cursor > 0:  # ctrl+backspace
            i = cursor - 1
            while i > 0 and text[i-1] == ' ': i -= 1
            while i > 0 and text[i-1] != ' ': i -= 1
            text = text[:i] + text[cursor:]
            cursor = i
        # submit
        if inpt.consume_enter() and text:
            submitted = text
            text = ""
            cursor = 0
            on_submit(submitted)

        # blinking-cursor
        blinking_cursor = "█" if (math.floor(time.time()*2) % 2 == 0) else " "
        display = text[:cursor] + blinking_cursor + text[cursor:]
        return Panel(f"[red]>[/red] {display}", style="white")

    return draw

@dataclass
class ContextInfo:
    name: str
    model: str = "opus-4.5"
    llm_current_output: str = ""
    llm_currently_running: bool = False
    last_llm_time: float = 0
    tokens: int = 32000
    max_tokens: int = 200000
    cost: float = 0.15
    messages: list = field(default_factory=list)
    input_stack: list = field(default_factory=list)

    def __post_init__(self):
        def on_submit(t):
            if t.startswith("/"): dispatch_command(t)
            else: self.call(t)
        self.input_stack = [make_input(on_submit)]
        state.contexts.add(self)
    
    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other
    
    def call(self, text):
        self.messages.append({"role": "user", "content": text})
        self.llm_currently_running = True
        self.llm_current_output = ""

        def run():
            for token in mock_llm_stream():
                self.llm_current_output += token
            self.messages.append({"role": "assistant", "content": self.llm_current_output})
            self.llm_currently_running = False
            self.last_llm_time = time.time()

        threading.Thread(target=run, daemon=True).start()

    def fork(self) -> ContextInfo:
        cpy = copy.copy(self)
        cpy.messages = copy.deepcopy(self.messages)
        cpy.input_stack = []
        cpy.__post_init__()  # fresh input handlers
        return cpy

    def push_ui(self, draw_fn):
        self.input_stack.append(draw_fn)


def get_content(msg: dict[str, str|Callable[[ContextInfo],str]], ctx: ContextInfo) -> str:
    c = msg["content"]
    return c(ctx) if callable(c) else c


@dataclass
class AppState:
    current_context: Optional['ContextInfo'] = None
    mode: str = "selection"
    contexts: set = field(default_factory=set)
    selection_input: callable = field(default_factory=lambda: make_input(lambda t: dispatch_command(t) if t.startswith("/") else None))

state = AppState()


ContextInfo("ctx1", messages=[
    {"role": "system", "content": "You are helpful.", "name": "sys-prompt-1"},
    {"role": "user", "content": "hello"},
])
ContextInfo("ctx2")
ContextInfo("debug_ctx", tokens=5000, messages=[
    {"role": "system", "content": "Debug mode."},
    {"role": "user", "content": "test input"},
])
ContextInfo("foobar", model="sonnet-4", tokens=45000, cost=0.08)



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
        self._keys = keys  # list of (character, key_name) tuples

    def consume(self, key: str) -> bool:
        '''Consume first event matching key name.'''
        for i, (char, key_name) in enumerate(self._keys):
            if key_name == key:
                self._keys.pop(i)
                return True
        return False

    def consume_text(self) -> str:
        '''Consume all printable characters.'''
        text = ""
        remaining = []
        for char, key_name in self._keys:
            if char and len(char) == 1 and char.isprintable():
                text += char
            else:
                remaining.append((char, key_name))
        self._keys[:] = remaining
        return text

    def consume_enter(self) -> bool:
        return self.consume("enter")
    def consume_backspace(self) -> bool:
        return self.consume("backspace")
    def consume_right(self) -> bool:
        return self.consume("right")
    def consume_up(self) -> bool:
        return self.consume("up")
    def consume_down(self) -> bool:
        return self.consume("down")



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

SPINNER = "/-\\|/-\\||"


def render_selection_left(inpt):
    ctxs = sorted(state.contexts, key=lambda c: c.name)
    idx = next((i for i, c in enumerate(ctxs) if c is state.current_context), 0)

    if inpt.consume_up() and idx > 0:
        state.current_context = ctxs[idx - 1]
        idx -= 1
    if inpt.consume_down() and idx < len(ctxs) - 1:
        state.current_context = ctxs[idx + 1]
        idx += 1
    if inpt.consume_enter() and ctxs:
        state.mode = "work"

    spin_char = SPINNER[int(time.time() * 10) % len(SPINNER)]
    now = time.time()
    lines = Text()
    for i, ctx in enumerate(ctxs):
        prefix = Text(">>  " if i == idx else "  ", style="red bold" if i == idx else "")
        spin = f" {spin_char}" if ctx.llm_currently_running else ""
        toks = f" ({ctx.tokens//1000}k)"
        # color: yellow=running, white=recent, dim=stale
        if ctx.llm_currently_running:
            color = "yellow"
        elif now - ctx.last_llm_time < 360:
            color = "white"
        else:
            color = "dim"
        style = f"bold {color}" if i == idx else color
        lines.append(prefix)
        lines.append(f"{ctx.name}{toks}{spin}\n", style=style)
    return Panel(lines, title="Contexts")



def render_selection_right():
    ctx = state.current_context
    if not ctx:
        return Panel("No contexts", title="Info")

    ratio = ctx.tokens / ctx.max_tokens
    bar_len = 20
    filled = int(ratio * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    info = Text()
    info.append(f"{ctx.name}  ", style="bold")
    info.append(f"{ctx.model}\n", style="dim")
    info.append(f"{bar} {ctx.tokens//1000}k/{ctx.max_tokens//1000}k\n", style="cyan")
    info.append(f"${ctx.cost:.2f}\n\n", style="dim")

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



def render_work_mode(ctx, inpt):
    conv = Text()
    for msg in ctx.messages:
        role = msg["role"]
        content = get_content(msg, ctx)
        if role == "user":
            conv.append(f"{content}\n", style="bold cyan")
        elif role == "assistant":
            conv.append(f"{content}\n", style="white")
        else:
            conv.append(f"{content}\n", style="dim")
    if ctx and ctx.llm_currently_running:
        conv.append(f"{ctx.llm_current_output}_\n", style="yellow")
    return conv


class Ex6App(App):
    CSS = "Screen { layout: vertical; } #main { height: 1fr; } #input { height: 4; }"

    def __init__(self):
        super().__init__()
        self._keys = []

    def compose(self) -> ComposeResult:
        yield Static(id="main")
        yield Static(id="input")

    def on_mount(self):
        self.set_interval(1/90, self._render_frame)

    def on_key(self, event):
        if event.key == "ctrl+c":
            self.exit()
            return
        self._keys.append((event.character, event.key))
        self._render_frame()

    def _render_frame(self):
        inpt = InputPass(self._keys)
        self._keys = []
        if state.mode == "selection":
            main = Layout()
            main.split_row(
                Layout(render_selection_left(inpt), name="left", ratio=1),
                Layout(render_selection_right(), name="right", ratio=2),
            )
            self.query_one("#main", Static).update(main)
        else:
            ctx = state.current_context
            if inpt.consume("escape"):
                state.mode = "selection"
            if ctx and ctx.messages:
                conv = render_work_mode(ctx, inpt)
            else:
                conv = Text("(empty conversation)\n", style="dim")
            self.query_one("#main", Static).update(Panel(conv, title=ctx.name if ctx else "Work"))
        self.query_one("#input", Static).update(render_input_box(inpt))


if __name__ == "__main__":
    load_plugins()
    Ex6App().run()




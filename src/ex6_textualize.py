from typing import Optional, Callable

import threading
import time
import copy
import glob
import os
import inspect
from dataclasses import dataclass, field
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Static, Input
from textual.containers import Container, Horizontal
from textual.message import Message


_commands = {}
_tools = {}


def mock_llm_stream():
    """Mock LLM data stream."""
    for _ in range(60):
        time.sleep(0.1)
        yield "token "


# --- Retained-Mode Widgets ---

class CommandInput(Input):
    """Text input at bottom of screen."""
    def __init__(self, on_submit: Callable[[str], None], **kwargs):
        super().__init__(placeholder="> ", **kwargs)
        self._on_submit = on_submit

    def on_input_submitted(self, event):
        text = self.value
        self.value = ""
        if text:
            self._on_submit(text)


class ContextList(Static):
    """Left panel in selection mode - list of contexts."""

    def on_mount(self):
        self.set_interval(0.1, self.refresh)

    def render(self):
        SPINNER = "/-\\|"
        spin_char = SPINNER[int(time.time() * 10) % len(SPINNER)]
        now = time.time()

        ctxs = sorted(state.contexts, key=lambda c: c.name)
        idx = next((i for i, c in enumerate(ctxs) if c is state.current_context), 0)

        lines = Text()
        for i, ctx in enumerate(ctxs):
            prefix = Text(">>  " if i == idx else "    ", style="red bold" if i == idx else "")
            spin = f" {spin_char}" if ctx.llm_currently_running else ""
            toks = f" ({ctx.tokens//1000}k)"

            if ctx.llm_currently_running:
                color = "yellow"
            elif now - ctx.last_llm_time < 360:
                color = "white"
            else:
                color = "dim"
            style = f"bold {color}" if i == idx else color

            lines.append(prefix)
            lines.append(f"{ctx.name}{toks}{spin}\n", style=style)

        return lines


class ContextInfoPanel(Static):
    """Right panel in selection mode - info about selected context."""

    def on_mount(self):
        self.set_interval(0.2, self.refresh)

    def render(self):
        ctx = state.current_context
        if not ctx:
            return Text("No contexts", style="dim")

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
                content = get_content(msg, ctx)
                toks = len(content) * 4
                info.append(f"{name} ({toks//1000}k)\n")
        else:
            info.append("(no messages)\n", style="dim")

        return info


class SelectionView(Container):
    """Selection mode view - left/right split."""

    def compose(self):
        with Horizontal():
            yield ContextList(id="ctx-list")
            yield ContextInfoPanel(id="ctx-info")

    def handle_key(self, event) -> bool:
        ctxs = sorted(state.contexts, key=lambda c: c.name)
        if not ctxs:
            return False
        idx = next((i for i, c in enumerate(ctxs) if c is state.current_context), 0)

        if event.key == "up" and idx > 0:
            state.current_context = ctxs[idx - 1]
            return True
        if event.key == "down" and idx < len(ctxs) - 1:
            state.current_context = ctxs[idx + 1]
            return True
        if event.key == "enter":
            self.app.show_work_mode()
            return True
        return False


class ConversationLog(Static):
    """Displays conversation messages in work mode."""

    def __init__(self, ctx, **kwargs):
        super().__init__(**kwargs)
        self.ctx = ctx

    def on_mount(self):
        self.set_interval(0.1, self.refresh)

    def render(self):
        conv = Text()
        for msg in self.ctx.messages:
            role = msg["role"]
            content = get_content(msg, self.ctx)
            if role == "user":
                conv.append(f"{content}\n", style="bold cyan")
            elif role == "assistant":
                conv.append(f"{content}\n", style="white")
            else:
                conv.append(f"{content}\n", style="dim")

        if self.ctx.llm_currently_running:
            conv.append(f"{self.ctx.llm_current_output}_\n", style="yellow")

        if not conv.plain:
            conv.append("(empty conversation)\n", style="dim")

        return conv


class WorkView(Container):
    """Work mode view - conversation display."""

    def __init__(self, ctx, **kwargs):
        super().__init__(**kwargs)
        self.ctx = ctx

    def compose(self):
        yield ConversationLog(self.ctx, id="conv-log")

    def handle_key(self, event) -> bool:
        if event.key == "escape":
            self.app.show_selection_mode()
            return True
        return False


# --- Data Classes ---

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
    widget_stack: list = field(default_factory=list)

    def __post_init__(self):
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

    def fork(self) -> 'ContextInfo':
        cpy = copy.copy(self)
        cpy.messages = copy.deepcopy(self.messages)
        cpy.widget_stack = []
        cpy.__post_init__()
        return cpy

    def push_widget(self, widget):
        self.widget_stack.append(widget)


def get_content(msg: dict, ctx: ContextInfo) -> str:
    c = msg["content"]
    return c(ctx) if callable(c) else c


@dataclass
class AppState:
    current_context: Optional['ContextInfo'] = None
    mode: str = "selection"
    contexts: set = field(default_factory=set)

state = AppState()


# Sample contexts
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


# --- Commands/Tools ---

def get_fn_name(fn):
    return fn.__name__


def command(fn):
    name = get_fn_name(fn)
    sig = inspect.signature(fn)
    spec = [(p.name, p.annotation if p.annotation != inspect.Parameter.empty else str)
            for p in sig.parameters.values()]
    _commands[name] = (fn, spec)
    return fn


def tool(fn):
    name = get_fn_name(fn)
    sig = inspect.signature(fn)
    spec = [(p.name, p.annotation if p.annotation != inspect.Parameter.empty else str)
            for p in sig.parameters.values()]
    _tools[name] = (fn, spec)
    return fn


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


# --- Main App ---

class Ex6App(App):
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    #input { height: 3; }
    Horizontal { height: 100%; }
    #ctx-list { width: 1fr; border: solid green; }
    #ctx-info { width: 2fr; border: solid blue; }
    #conv-log { height: 100%; border: solid cyan; }
    """

    def compose(self) -> ComposeResult:
        yield Container(id="main")
        yield CommandInput(self._on_submit, id="input")

    def on_mount(self):
        self.show_selection_mode()
        self.query_one("#input").focus()

    def _on_submit(self, text: str):
        if text.startswith("/"):
            dispatch_command(text)
        elif state.mode == "work" and state.current_context:
            state.current_context.call(text)

    def show_selection_mode(self):
        state.mode = "selection"
        main = self.query_one("#main")
        main.remove_children()
        main.mount(SelectionView(id="selection-view"))

    def show_work_mode(self):
        if not state.current_context:
            return
        state.mode = "work"
        main = self.query_one("#main")
        main.remove_children()
        main.mount(WorkView(state.current_context, id="work-view"))

    def on_key(self, event):
        if event.key == "ctrl+c":
            self.exit()
            return

        # Propagate to widget stack first
        ctx = state.current_context
        if ctx and ctx.widget_stack:
            for widget in reversed(ctx.widget_stack):
                if hasattr(widget, 'handle_key') and widget.handle_key(event):
                    return

        # Then to current view
        if state.mode == "selection":
            view = self.query_one("#selection-view", SelectionView)
            if view.handle_key(event):
                return
        elif state.mode == "work":
            view = self.query_one("#work-view", WorkView)
            if view.handle_key(event):
                return


if __name__ == "__main__":
    load_plugins()
    Ex6App().run()


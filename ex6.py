import threading
import time
import shlex
import glob
import os
from dataclasses import dataclass, field
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
import readchar


# =============================================================================
# REGISTRIES
# =============================================================================

_hooks = {
    "on_startup": [],
    "on_key": [],
    "on_submit": [],
    "on_tick": [],
    "on_render_status": [],
}

_commands = {}  # name -> (callback, args_spec)


# =============================================================================
# HOOK DECORATOR (namespaced)
# =============================================================================

class _HookNamespace:
    def _make_decorator(self, event_name):
        def decorator(fn):
            _hooks[event_name].append(fn)
            return fn
        return decorator

    @property
    def on_startup(self):
        return self._make_decorator("on_startup")

    @property
    def on_key(self):
        return self._make_decorator("on_key")

    @property
    def on_submit(self):
        return self._make_decorator("on_submit")

    @property
    def on_tick(self):
        return self._make_decorator("on_tick")

    @property
    def on_render_status(self):
        return self._make_decorator("on_render_status")


hook = _HookNamespace()


# =============================================================================
# COMMAND DECORATOR
# =============================================================================

def command(name, args=None):
    """
    @command("echo", args=[("text", str)])
    def echo_cmd(text):
        ...
    """
    args = args or []
    def decorator(fn):
        _commands[name] = (fn, args)
        return fn
    return decorator


# =============================================================================
# DISPATCH
# =============================================================================

def dispatch(event_name, *args, **kwargs):
    """Call all hooks for event. If any returns True, stop propagation."""
    for fn in _hooks.get(event_name, []):
        result = fn(*args, **kwargs)
        if result is True:
            return True
    return False


def dispatch_command(text):
    """Parse and dispatch a /command. Returns True if handled."""
    if not text.startswith("/"):
        return False

    try:
        parts = shlex.split(text[1:])  # remove leading /
    except ValueError:
        parts = text[1:].split()

    if not parts:
        return False

    cmd_name = parts[0]
    raw_args = parts[1:]

    if cmd_name not in _commands:
        console.add(f"Unknown command: /{cmd_name}\n")
        return True

    fn, args_spec = _commands[cmd_name]

    # Parse typed args
    parsed = []
    for i, (arg_name, arg_type) in enumerate(args_spec):
        if i < len(raw_args):
            try:
                parsed.append(arg_type(raw_args[i]))
            except (ValueError, TypeError):
                console.add(f"Invalid arg {arg_name}: expected {arg_type.__name__}\n")
                return True
        else:
            console.add(f"Missing arg: {arg_name}\n")
            return True

    return fn(*parsed)


# =============================================================================
# CONSOLE
# =============================================================================

class Console:
    def __init__(self):
        self._lines = []
        self._current = ""
        self._lock = threading.Lock()

    def add(self, text):
        with self._lock:
            for char in text:
                if char == "\n":
                    self._lines.append(self._current)
                    self._current = ""
                else:
                    self._current += char

    def clear(self):
        with self._lock:
            self._lines = []
            self._current = ""

    def render(self):
        with self._lock:
            lines = self._lines.copy()
            current = self._current

        content = "\n".join(lines)
        if current:
            content += ("\n" if lines else "") + current
        return content or " "


console = Console()


# =============================================================================
# LLM (dummy)
# =============================================================================

class LLM:
    def __init__(self):
        self.busy = False
        self.response = ""
        self._pending = ""
        self._lock = threading.Lock()

    def send(self, prompt):
        """Start async generation (dummy: simulates streaming)."""
        self.busy = True
        self.response = ""
        threading.Thread(target=self._generate, args=(prompt,), daemon=True).start()

    def _generate(self, prompt):
        # Dummy: echo back slowly
        reply = f"You said: {prompt}"
        for char in reply:
            time.sleep(0.05)
            with self._lock:
                self._pending += char
        with self._lock:
            self._pending += "\n"
        self.busy = False

    def has_new_tokens(self):
        with self._lock:
            return len(self._pending) > 0

    def consume_tokens(self):
        with self._lock:
            tokens = self._pending
            self._pending = ""
            self.response += tokens
            return tokens


# =============================================================================
# STATE
# =============================================================================

@dataclass
class AppState:
    input_buffer: str = ""
    current_window: list = field(default_factory=list)
    llm: LLM = field(default_factory=LLM)
    running: bool = True


class ThreadSafeState:
    def __init__(self, state):
        self._state = state
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self._state

    def __exit__(self, *args):
        self._lock.release()


state = ThreadSafeState(AppState())


# =============================================================================
# PLUGIN LOADER
# =============================================================================

def load_plugins():
    plugin_dir = os.path.join(os.path.dirname(__file__) or ".", ".ex6")
    if not os.path.isdir(plugin_dir):
        return

    for path in glob.glob(os.path.join(plugin_dir, "*.py")):
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, path, "exec"), {"__name__": "__plugin__", "__file__": path})


# =============================================================================
# INPUT THREAD
# =============================================================================

def input_thread():
    with state as s:
        running = s.running

    while running:
        key = readchar.readkey()

        with state as s:
            if key == readchar.key.CTRL_C:
                s.running = False
            elif key == readchar.key.BACKSPACE:
                s.input_buffer = s.input_buffer[:-1]
            elif key == readchar.key.ENTER:
                text = s.input_buffer.strip()
                s.input_buffer = ""
                if text:
                    # Check for command first
                    if text.startswith("/"):
                        dispatch_command(text)
                    else:
                        dispatch("on_submit", text, s)
            elif len(key) == 1 and ord(key) >= 32:
                # Dispatch on_key, if not handled, add to buffer
                if not dispatch("on_key", key, s):
                    s.input_buffer += key
            else:
                # Special keys
                dispatch("on_key", key, s)

            running = s.running


# =============================================================================
# RENDER
# =============================================================================

def render():
    with state as s:
        input_buffer = s.input_buffer
        llm = s.llm

    # Get status from hooks
    status_parts = []
    for fn in _hooks["on_render_status"]:
        with state as s:
            result = fn(s)
        if result:
            status_parts.append(result)
    status = " | ".join(status_parts) if status_parts else ""

    layout = Layout()

    parts = [
        Layout(Panel(console.render(), title="Console", border_style="dim"), name="console"),
    ]

    if status:
        parts.append(Layout(Panel(status, border_style="yellow"), name="status", size=3))

    parts.append(Layout(Panel(f"> {input_buffer}█", title="Input", border_style="blue"), name="input", size=3))

    layout.split_column(*parts)
    return layout


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    load_plugins()
    dispatch("on_startup")

    threading.Thread(target=input_thread, daemon=True).start()

    with Live(render(), screen=True, auto_refresh=False) as live:
        with state as s:
            running = s.running

        while running:
            with state as s:
                dispatch("on_tick", s)

            live.update(render(), refresh=True)

            with state as s:
                running = s.running

            time.sleep(0.016)  # ~60fps

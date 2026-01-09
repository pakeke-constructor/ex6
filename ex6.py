
import threading
import time
import shlex
import glob
import os
from dataclasses import dataclass, field
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
import readchar



_hooks = {"on_startup": [], "on_key": [], "on_submit": [], "on_tick": [], "on_render_status": []}
_commands = {}



class _HookNamespace:
    def _make(self, name):
        def dec(fn):
            _hooks[name].append(fn)
            return fn
        return dec

    @property
    def on_startup(self): return self._make("on_startup")
    @property
    def on_key(self): return self._make("on_key")
    @property
    def on_submit(self): return self._make("on_submit")
    @property
    def on_tick(self): return self._make("on_tick")
    @property
    def on_render_status(self): return self._make("on_render_status")

hook = _HookNamespace()


def command(name, args=None):
    args = args or []
    def dec(fn):
        _commands[name] = (fn, args)
        return fn
    return dec



@dataclass
class AppState:
    input_buffer: str = ""
    console: list = field(default_factory=list)  # list of strings
    current_window: list = field(default_factory=list)
    running: bool = True
    keys_this_frame: list = field(default_factory=list)  # keys pressed since last tick
    input_ui: object = None  # Rich renderable for custom input UI, or None


class ThreadSafeState:
    def __init__(self, state):
        self._state = state
        self._lock = threading.Lock()
    def __enter__(self):
        self._lock.acquire()
        return self._state
    def __exit__(self, *a):
        self._lock.release()

state = ThreadSafeState(AppState())



def dispatch(event, *args):
    for fn in _hooks.get(event, []):
        if fn(*args) is True:
            return True
    return False


def dispatch_command(text):
    if not text.startswith("/"):
        return False
    try:
        parts = shlex.split(text[1:])
    except:
        parts = text[1:].split()
    if not parts:
        return False

    name, raw = parts[0], parts[1:]
    if name not in _commands:
        with state as s:
            s.console.append(f"Unknown command: /{name}")
        return True

    fn, spec = _commands[name]
    parsed = []
    for i, (arg_name, arg_type) in enumerate(spec):
        if i < len(raw):
            try:
                parsed.append(arg_type(raw[i]))
            except:
                with state as s:
                    s.console.append(f"Invalid arg {arg_name}")
                return True
        else:
            with state as s:
                s.console.append(f"Missing arg: {arg_name}")
            return True
    return fn(*parsed)



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
        with state as s:
            s.keys_this_frame.append(key)  # Always collect keys

            if key == readchar.key.CTRL_C:
                s.running = False
            elif s.input_ui is not None:
                pass  # Plugin handles keys in on_tick
            elif key == readchar.key.BACKSPACE:
                s.input_buffer = s.input_buffer[:-1]
            elif key == readchar.key.ENTER:
                text = s.input_buffer.strip()
                s.input_buffer = ""
                if text:
                    if text.startswith("/"):
                        dispatch_command(text)
                    else:
                        dispatch("on_submit", text, s)
            elif len(key) == 1 and ord(key) >= 32:
                if not dispatch("on_key", key, s):
                    s.input_buffer += key
            else:
                dispatch("on_key", key, s)

            if not s.running:
                break



def render():
    with state as s:
        console_text = "\n".join(s.console) or " "
        input_buffer = s.input_buffer
        input_ui = s.input_ui

        status_parts = [fn(s) for fn in _hooks["on_render_status"]]

    status = " | ".join(p for p in status_parts if p)

    layout = Layout()
    parts = [Layout(Panel(console_text, title="Console", border_style="dim"), name="console")]
    if status:
        parts.append(Layout(Panel(status, border_style="yellow"), name="status", size=3))
    if input_ui is not None:
        parts.append(Layout(input_ui, name="input"))  # Dynamic size
    else:
        parts.append(Layout(Panel(f"> {input_buffer}█", title="Input", border_style="blue"), name="input", size=3))
    layout.split_column(*parts)
    return layout



if __name__ == "__main__":
    load_plugins()
    dispatch("on_startup")
    threading.Thread(target=input_thread, daemon=True).start()

    with Live(render(), screen=True, auto_refresh=False) as live:
        while True:
            with state as s:
                s.keys_this_frame.clear()  # Clear at start of tick
            time.sleep(0.016)  # Allow keys to accumulate
            with state as s:
                dispatch("on_tick", s)
                if not s.running:
                    break
            live.update(render(), refresh=True)

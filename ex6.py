
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



_commands = {}



@dataclass
class AppState:
    input_buffer: str = ""
    console: list = field(default_factory=list)  # list of strings
    current_window: list = field(default_factory=list)
    running: bool = True
    keys_this_frame: list = field(default_factory=list)  # keys pressed since last tick
    input_ui: object = None  # Rich renderable for custom input UI, or None



def command(fn):
    '''
    used like:

    @ex6.command
    def my_command(arg1, arg2): pass
        
    now, `/command a b` should be valid command
    '''
    name = get_fn_name(fn)
    _commands[name] = fn
    return fn


def tool(fn):
    '''
    @ex6.tool
    def my_llm_tool(arg1, arg2):
        pass
        
    can be included in ctx windows for LLMs.
    '''
    name = get_fn_name(fn)
    _commands[name] = fn
    return fn


class InputPass:
    '''
    A new InputPass object is created every frame.
    records keyboard activity.
    '''
    def __init__(self):
        pass# TODO.

    def consume(self, key: str) -> bool:
        # TODO: consumes `key` for this frame.
        # returns True iff this key was pressed.
        # False if this key wasnt pressed.
        # (When consumed, future calls of `consume` will return false for this frame)
        return True

    def consume_text(self) -> str:
        # TODO: consumes any text for this frame.
        # future calls of `consume_text` and consume(?) will return False for this frame

        txt = "a" # the current key that is pressed
        return txt

    def emit_keypress(self, key):
        # TODO: consumes key-press for this frame.
        # future calls of `consume` will return false
        pass 

    def consume_enter(self) -> bool:
        return self.consume(readchar.key.ENTER)

    def consume_backspace(self) -> bool:
        return self.consume(readchar.key.BACKSPACE)

    def consume_left(self) -> int:
        return self.consume(readchar.key.LEFT)
    def consume_right(self) -> int:
        return self.consume(readchar.key.RIGHT)
    def consume_up(self) -> int:
        return self.consume(readchar.key.UP)
    def consume_down(self) -> int:
        return self.consume(readchar.key.DOWN)



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
                pass # send to 

            if not s.running:
                break



def render():
    pass


if __name__ == "__main__":
    load_plugins()
    threading.Thread(target=input_thread, daemon=True).start()

    with Live(render(), screen=True, auto_refresh=False) as live:
        while True:
            with state as s:
                s.keys_this_frame.clear()  # Clear at start of tick
            time.sleep(0.016)  # Allow keys to accumulate
            live.update(render(), refresh=True)




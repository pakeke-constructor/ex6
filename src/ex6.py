
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
import readchar



_commands = {}

_tools = {}



def get_fn_name(fn):
    return fn.__name__

@dataclass
class AppState:
    input_buffer: str = ""
    console: list = field(default_factory=list)
    current_window: list = field(default_factory=list)
    keys_this_frame: list = field(default_factory=list)
    input_stack: list = field(default_factory=list)



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
            s.keys_this_frame.append(key)

def push_input(draw_fn):
    with state as s:
        s.input_stack.append(draw_fn)


class Context:
    def __init__(self, name, messages):
        self.name = name
        self.messages = messages


_current_ctx: Optional[Context] = None

def set_context(ctx): _current_ctx = ctx
def get_context(): return _current_ctx



def _render():
    # TODO: do layout properly.
    return Layout()


def tick() -> bool:
    with state as s:
        keys = list(s.keys_this_frame)
        s.keys_this_frame.clear()

    inpt = InputPass(keys)

    if inpt.consume(readchar.key.CTRL_C):
        return False

    with state as s:
        if s.input_stack:
            if s.input_stack[-1](inpt) is None:
                s.input_stack.pop()
    return True


if __name__ == "__main__":
    load_plugins()
    threading.Thread(target=input_thread, daemon=True).start()

    with Live(_render(), screen=True, auto_refresh=False) as live:
        while tick():
            time.sleep(1/60)
            live.update(_render(), refresh=True)




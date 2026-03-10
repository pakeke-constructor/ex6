
import os
import sys
import hashlib
from pathlib import Path
from collections import deque
from datetime import datetime

os.environ.setdefault('ESCDELAY', '25')  # reduce escape key delay (ms)
# (if using SSH, you might want to set this higher. Ask some LLM to explain why.)

sys.modules['ex6'] = sys.modules[__name__]  # so plugins can `import ex6`

# Debug ring buffer
_debug_buffer = deque(maxlen=1000)

def debug_print(*args, **kwargs):
    ts = datetime.now().strftime("%H:%M:%S")
    msg = " ".join(str(a) for a in args)
    _debug_buffer.append(f"[{ts}] {msg}")


_fatal_error = None

import threading
def _thread_excepthook(args):
    # args: ExceptHookArgs(exc_type, exc_value, exc_traceback, thread)
    import traceback
    tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    debug_print(f"THREAD CRASH [{args.thread}]:\n{tb}")
    global _fatal_error
    _fatal_error = (f"{args.exc_type.__name__}: {args.exc_value}")
threading.excepthook = _thread_excepthook

from blessed import Terminal
from typing import Union, Tuple, List, Optional, Literal, Callable
import time
from dataclasses import dataclass, field
from typing import Optional,Any
import threading
import inspect
from typing import get_origin, get_args
import copy
import time
import glob



_commands = {}
_output_renderers = []

# Type aliases for output rendering
RenderFn = Callable[['ScreenBuffer', int, int, int], int]  # fn(buf, x, y, w) -> rows
OutputLine = Union[str, RenderFn]  # str or render fn
OutputRendererFn = Callable[[str, list, 'Context'], None]  # fn(role, lines, ctx) -> None

def output_renderer(fn: OutputRendererFn) -> OutputRendererFn:
    '''
    Called once per message. Mutate `lines` in-place (replace str with RenderFn).

    @ex6.output_renderer
    def syntax_highlighting(role: str, lines: list[ex6.OutputLine], ctx: ex6.Context) -> None:
        ...
    '''
    _output_renderers.append(fn)
    return fn




def command(fn):
    '''
    used like:

    @ex6.command
    def my_command(arg1, arg2): pass

    now, `/command a b` should be valid command
    '''
    name = fn.__name__
    sig = inspect.signature(fn)
    spec = [(p.name, p.annotation if p.annotation != inspect.Parameter.empty else str)
            for p in sig.parameters.values()]
    _commands[name] = (fn, spec)
    return fn





def _coerce_arg(value: str, typ):
    """Convert string arg to the annotated type."""
    origin = get_origin(typ)
    # Handle Optional[X] / Union[X, None]
    if origin is Union:
        args = [t for t in get_args(typ) if t is not type(None)]
        typ = args[0] if len(args) == 1 else str
    # Handle basic types
    if typ in (str, inspect.Parameter.empty):
        return value
    return typ(value)

def dispatch_command(text: str):
    if not text.startswith("/"): return False
    parts = text[1:].split()
    if not parts: return False

    name, args = parts[0], parts[1:]
    if name not in _commands: return True

    fn, spec = _commands[name]
    parsed = []
    for i, (_, typ) in enumerate(spec):
        if i < len(args):
            parsed.append(_coerce_arg(args[i], typ))
        else:
            parsed.append(None)
    return fn(*parsed)


@command
def debug():
    enter_scroll_mode()
    print("="*30)
    print("DEBUG LOG")
    print("="*30)
    print(f"({len(_debug_buffer)} entries)\n")
    for line in _debug_buffer:
        print(line)

@command
def context():
    enter_scroll_mode()
    ctx = state.current
    if not ctx:
        print("No active context."); return
    print("\n\n\n\n\n")
    print(f"=== Context: {ctx.name} ({ctx.model}) ===\n")
    for i, msg in enumerate(ctx.messages):
        print(f"--- [{i}] {msg.role} ---")
        print(msg.get_msg(ctx))
        print()

_log_keys = False
@command
def keys():
    """Toggle key debug logging."""
    global _log_keys
    _log_keys = not _log_keys
    debug_print(f"Key logging {'ON' if _log_keys else 'OFF'}")



def get_folder() -> Path:
    """Returns app data folder: %APPDATA%/ex6 on Windows, ~/.ex6 on Unix."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home()
    folder = base / "ex6"
    folder.mkdir(exist_ok=True)
    return folder




OVERRIDES = {}
_OVERRIDDEN = set()

def overridable(fn):
    OVERRIDES[fn.__name__] = fn
    def wrap_fn(*a, **ka):
        return OVERRIDES[fn.__name__](*a, **ka)
    return wrap_fn

def override(fn):
    name = fn.__name__
    if name not in OVERRIDES:
        raise RuntimeError(f"'{name}' not overridable")
    if name in _OVERRIDDEN:
        raise RuntimeError(f"'{name}' already overridden")
    _OVERRIDDEN.add(name)
    OVERRIDES[name] = fn
    return fn



@dataclass
class AppState:
    contexts: dict[str,Context] = field(default_factory=dict)
    current: 'Context' = None  # pyright: ignore - always valid when contexts is non-empty
    mode: Literal["selection", "work", "help", "scroll"] = "selection"
    _prev_mode: Literal["selection", "work", "help", "scroll"] = "selection"  # for restoring after scroll
    term: 'Terminal' = None  # set in main


state = AppState()

def enter_scroll_mode():
    """Exit fullscreen for scroll mode. Caller prints, main loop handles re-entry."""
    if state.mode == "scroll": return
    state._prev_mode = state.mode
    state.mode = "scroll"
    print(state.term.exit_fullscreen, end='', flush=True)



@overridable
def invoke_llm(ctx):
    """Override this to use real LLM."""
    for _ in range(60):
        time.sleep(0.1)
        yield ResponseChunk("text", "token ", 1)


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, Callable[['Context'], str]]
    tools: dict[str, Callable] = field(default_factory=dict)
    chunks: Optional[list] = None  # ordered ResponseChunks (for assistant msgs)
    tool_calls: Optional[list] = None  # for assistant msgs with tool calls
    tool_call_id: Optional[str] = None  # for tool result msgs

    def get_msg(self, ctx: 'Context'):
        c = self.content
        return c(ctx) if callable(c) else c


@dataclass
class ResponseChunk:
    type: str  # "text", "cot", "tool"
    content: str = ""
    tokens: int = 1  # for cot

@dataclass
class LLMResult:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    error: Optional[str] = None
    cost: Optional[float] = None


def _ensure_unique_name(name):
    if name not in state.contexts:
        return name
    # otherwise, search for new name
    while True:
        last = name[-1]
        if last.isdigit() and last != '9': name = name[:-1] + str(int(last) + 1)
        elif last == '9': name = name + '0'
        else: name = name + '1'
        if name not in state.contexts: break
    return name



def _check_tool_args(fn: Callable, args: dict) -> dict:
    """Validate args against fn annotations, raise TypeError if mismatch."""
    hints = fn.__annotations__
    for name, val in args.items():
        if name in hints and not isinstance(val, hints[name]):
            raise TypeError(f"{fn.__name__}: arg '{name}' expected {hints[name].__name__}, got {type(val).__name__}")
    return args


_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array"}


def _validate_tool_sig(name: str, fn: Callable):
    """Ensure tool has (ctx: Context, ...) signature."""
    params = list(inspect.signature(fn).parameters.values())
    if len(params) < 1:
        raise TypeError(f"Tool '{name}' must have at least 1 param: (ctx)")
    if params[0].annotation not in (Context, 'Context', inspect.Parameter.empty):
        raise TypeError(f"Tool '{name}' first param must be ctx: Context")


def tool_to_schema(name: str, fn: Callable) -> dict:
    _validate_tool_sig(name, fn)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())[1:]  # skip ctx
    props = {}
    required = []

    for param in params:
        pname = param.name
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        ptype = param.annotation if param.annotation != inspect.Parameter.empty else str
        if get_origin(ptype) is Union:
            ptype = [a for a in get_args(ptype) if a is not type(None)][0]
        props[pname] = {"type": _TYPE_MAP.get(ptype, "string")}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (fn.__doc__ or "").strip(),
            "parameters": {"type": "object", "properties": props, "required": required}
        }
    }



@overridable
def call_tools(ctx: Context, llm_result: LLMResult) -> bool:
    '''
    Blocks the thread until tools are complete.
    Then, returns boolean; whether the LLM should loop.
    (by default; the LLM loops when there are tool-calls.)

    This function can be overridden if you have a special way of calling tools.
    (e.g cloudflare's code-mode  https://blog.cloudflare.com/code-mode/ )
    '''
    if not llm_result.tool_calls:
        return False

    tools = ctx.get_tools()
    threads = []
    results = []
    for tc in llm_result.tool_calls:
        fn = tools.get(tc["name"])
        if not fn: continue
        result = {"id": tc["id"], "value": None}
        results.append(result)
        def run_tool(fn=fn, tc=tc, result=result):
            result["value"] = fn(ctx, **_check_tool_args(fn, tc["args"]))
        t = threading.Thread(target=run_tool)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    # Add tool results as messages
    for r in results:
        ctx.messages.append(Message(role="tool", content=str(r["value"] or ""), tool_call_id=r["id"]))
    return True



@dataclass
class Context:
    name: str
    model: str
    messages: list = field(default_factory=list)
    max_tokens: int = 200000
    llm_is_running: bool = False
    llm_current_output: list = field(default_factory=list)
    last_invoke_time_end: float = 0
    last_invoke_time_start: float = 0
    llm_result: Optional[LLMResult] = None
    input_stack: list = field(default_factory=list)
    _msg_lock: threading.Lock = field(default_factory=threading.Lock)
    llm_suspended: bool = False
    data: dict[str,Any] = field(default_factory=dict) # a dict for plugins to store stuff.
    _read_hashes: dict[str,str] = field(default_factory=dict) # file read tracking
    _prev_height: int = 0 # how many lines were used in rendering last frame

    def token_count(self) -> int:
        if self.llm_result:
            return self.llm_result.input_tokens + self.llm_result.output_tokens
        return sum(len(m.content) // 4 for m in self.messages if isinstance(m.content, str))

    def is_token_count_estimate(self) -> bool:
        "If no llmResult, then we are estimating the token-count via the //4 trick."
        return not self.llm_result

    def __post_init__(self):
        state.contexts[self.name] = self

    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other
    def is_running(self): return self.llm_is_running

    @staticmethod
    def _file_hash(path):
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def mark_file_read(self, path):
        self._read_hashes[os.path.abspath(path)] = self._file_hash(path)

    def has_read_file(self, path):
        p = os.path.abspath(path)
        return p in self._read_hashes and self._read_hashes[p] == self._file_hash(p)

    def get_read_files(self):
        return list(self._read_hashes.keys())

    def get_tools(self) -> dict[str, Callable]:
        tools = {}
        for m in self.messages:
            tools.update(m.tools)
        return tools

    def get_tool_schemas(self) -> list[dict]:
        return [tool_to_schema(name, fn) for name, fn in self.get_tools().items()]

    def invoke(self, text, llm_fn=None):
        llm_fn = llm_fn or invoke_llm
        self.messages.append(Message(role="user", content=text))
        self.llm_is_running = True

        def do_llm():
            self.last_invoke_time_start = time.time()
            self.llm_current_output = []
            for item in llm_fn(self):
                if isinstance(item, ResponseChunk):
                    self.llm_current_output.append(item)
                elif isinstance(item, LLMResult):
                    self.llm_result = item
            content = "".join(c.content for c in self.llm_current_output if c.type == "text")
            tool_calls = self.llm_result.tool_calls if self.llm_result else None
            self.messages.append(Message(role="assistant", content=content, chunks=list(self.llm_current_output), tool_calls=tool_calls))

        def run():
            should_loop = True
            while should_loop:
                do_llm()
                if not self.llm_result: break
                self.llm_suspended = True
                should_loop = call_tools(self, self.llm_result)
                self.llm_suspended = False
            self.llm_is_running = False
            self.last_invoke_time_end = time.time()

        threading.Thread(target=run, daemon=True).start()
    
    def fork(self, new_name: Optional[str] = None) -> 'Context':
        cpy = copy.copy(self)
        cpy.messages = copy.deepcopy(self.messages)
        cpy._read_hashes = dict(self._read_hashes)
        cpy.data = copy.copy(self.data)
        cpy.input_stack = []
        cpy.name = _ensure_unique_name(new_name or self.name)
        cpy.__post_init__()
        return cpy

    def push_ui(self, draw_fn):
        self.input_stack.append(draw_fn)









#====================================
# UI, rendering, input, main-loop:
#====================================


Rect = Tuple[int, int, int, int]  # (x, y, w, h)
RegionLike = Union['Region', Rect]


class ScreenBuffer:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.chars = [[' '] * w for _ in range(h)]
        self.styles: List[List[Optional[str]]] = [[None] * w for _ in range(h)]
        self.txt_colors: List[List[Optional[str]]] = [[None] * w for _ in range(h)]
        self.bg_colors: List[List[Optional[str]]] = [[None] * w for _ in range(h)]

    def put(self, x, y, char, style=None, txt_color=None, bg_color=None):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.chars[y][x] = char
            self.styles[y][x] = style
            self.txt_colors[y][x] = txt_color
            self.bg_colors[y][x] = bg_color

    def puts(self, x, y, text, style=None, txt_color=None, bg_color=None):
        for i, c in enumerate(text):
            self.put(x + i, y, c, style, txt_color, bg_color)

    def clear(self):
        for row in self.chars: row[:] = [' '] * self.w
        for row in self.styles: row[:] = [None] * self.w
        for row in self.txt_colors: row[:] = [None] * self.w
        for row in self.bg_colors: row[:] = [None] * self.w

    def flush(self, term):
        out = ""
        for y in range(self.h):
            out += term.move(y, 0)
            for x in range(self.w):
                c = self.chars[y][x]
                fg, s, bg = self.txt_colors[y][x], self.styles[y][x], self.bg_colors[y][x]
                parts = [p for p in [fg, s] if p]
                if bg: parts.append(f"on_{bg}")
                attr = "_".join(parts) if parts else None
                styled = getattr(term, attr, None) if attr else None
                out += styled(c) if styled else c
        print(out, end='', flush=True)

    def fill(self, r: Rect, char='█', style=None, txt_color=None, bg_color=None):
        x, y, w, h = r
        for row in range(y, y + h):
            for col in range(x, x + w):
                self.put(col, row, char, style, txt_color, bg_color)

    def rect_line(self, r: Rect, style=None, txt_color=None, bg_color=None):
        x, y, w, h = r
        if w < 2 or h < 2: return
        for col in range(x + 1, x + w - 1):
            self.put(col, y, '─', style, txt_color, bg_color)
            self.put(col, y + h - 1, '─', style, txt_color, bg_color)
        for row in range(y + 1, y + h - 1):
            self.put(x, row, '│', style, txt_color, bg_color)
            self.put(x + w - 1, row, '│', style, txt_color, bg_color)
        self.put(x, y, '┌', style, txt_color, bg_color)
        self.put(x + w - 1, y, '┐', style, txt_color, bg_color)
        self.put(x, y + h - 1, '└', style, txt_color, bg_color)
        self.put(x + w - 1, y + h - 1, '┘', style, txt_color, bg_color)

    def hline(self, r: Rect, style=None, txt_color=None, bg_color=None):
        x, y, w, _ = r
        for i in range(w):
            self.put(x + i, y, '─', style, txt_color, bg_color)

    def vline(self, r: Rect, style=None, txt_color=None, bg_color=None):
        x, y, _, h = r
        for i in range(h):
            self.put(x, y + i, '│', style, txt_color, bg_color)

    def print_wrapped(self, txt: str, x: int, y: int, w: int, style=None, txt_color=None, bg_color=None) -> int:
        """Write text with word-wrapping at width w. Returns number of lines consumed."""
        lines = txt.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        row = 0
        for line in lines:
            if not line:
                row += 1; continue
            col = 0
            words = line.split(' ')
            for wi, word in enumerate(words):
                if col > 0 and col + len(word) > w:
                    row += 1; col = 0
                for c in word:
                    if col >= w: row += 1; col = 0
                    self.put(x + col, y + row, c, style, txt_color, bg_color)
                    col += 1
                if wi < len(words) - 1 and col < w:
                    col += 1  # space
            row += 1
        return max(row, 1)

    def writer(self, x, y, w):
        return WrapWriter(self, x, y, w)

    def print_contained(self, txt: str, r: Rect, style=None, txt_color=None, bg_color=None, wrap=True, newlines=True) -> int:
        x, y, w, h = r
        if not newlines:
            txt = txt.replace('\n', ' ')
        txt = txt.replace('\r\n', '\n').replace('\r', '\n')
        row, col = 0, 0
        for c in txt:
            if c == '\n': row += 1; col = 0; continue
            if wrap and col >= w: row += 1; col = 0
            if row >= h or (not wrap and col >= w): continue
            self.put(x + col, y + row, c, style, txt_color, bg_color)
            col += 1
        return row + 1 if col > 0 or row == 0 else row




class WrapWriter:
    def __init__(self, buf, x, y, w):
        self.buf, self.x, self.y, self.w = buf, x, y, w
        self.cx, self.cy = 0, 0

    def put(self, ch, style=None, txt_color=None, bg_color=None):
        if self.cx >= self.w:
            self.cy += 1; self.cx = 0
        self.buf.put(self.x + self.cx, self.y + self.cy, ch, style, txt_color, bg_color)
        self.cx += 1

    def newline(self):
        self.cy += 1; self.cx = 0

    @property
    def lines(self):
        return self.cy + 1 if self.cx > 0 else max(self.cy, 1)


class Region(tuple):
    """
    A class reprenting a (x,y,w,h) area on the screen.
    Used for laying out ui.
    """
    def __new__(cls, x: int = 0, y: int = 0, w: int = 0, h: int = 0):
        return super().__new__(cls, (int(x), int(y), max(0, int(w)), max(0, int(h))))
    
    def __repr__(self):
        return f"Region{super().__repr__()}"
    
    def split_vertical(self, *ratios: float) -> List['Region']:
        norm = [r / sum(ratios) for r in ratios]
        regions = []
        accum_y = self[1]
        for ratio in norm:
            h = int(self[3] * ratio)
            regions.append(Region(self[0], accum_y, self[2], h))
            accum_y += h
        return regions
    
    def split_horizontal(self, *ratios: float) -> List['Region']:
        norm = [r / sum(ratios) for r in ratios]
        regions = []
        accum_x = self[0]
        for ratio in norm:
            w = int(self[2] * ratio)
            regions.append(Region(accum_x, self[1], w, self[3]))
            accum_x += w
        return regions
    
    def grid(self, cols: int, rows: int) -> List['Region']:
        cell_w = self[2] // cols
        cell_h = self[3] // rows
        regions = []
        for row in range(rows):
            for col in range(cols):
                x = self[0] + cell_w * col
                y = self[1] + cell_h * row
                regions.append(Region(x, y, cell_w, cell_h))
        return regions
    
    def shrink(self, left: int, top: Optional[int] = None, right: Optional[int] = None, bottom: Optional[int] = None) -> 'Region':
        top = top if top is not None else left
        right = right if right is not None else left
        bottom = bottom if bottom is not None else top
        return Region(
            self[0] + left,
            self[1] + top,
            self[2] - left - right,
            self[3] - top - bottom
        )
    
    def move(self, dx: int, dy: int) -> 'Region':
        return Region(self[0] + dx, self[1] + dy, self[2], self[3])




class InputPass:
    KEY_ALIASES = {'\x17': 'KEY_CTRL_BACKSPACE', '\x7f': 'KEY_CTRL_BACKSPACE', '\x1bd': 'KEY_CTRL_DELETE'}

    def __init__(self, keys: list):
        self._keys = list(keys)

    def consume(self, name: str) -> bool:
        for i, k in enumerate(self._keys):
            key_name = self.KEY_ALIASES.get(str(k), k.name)
            if key_name == name or k.name == name:
                self._keys.pop(i)
                return True
        return False

    def consume_text(self) -> str:
        text = ""
        remaining = []
        for k in self._keys:
            if not k.is_sequence and str(k).isprintable():
                text += str(k)
            else:
                remaining.append(k)
        self._keys[:] = remaining
        return text



@overridable
def make_input(on_submit):
    text, cursor = "", 0

    def prev_word(text, i):
        while i > 0 and not text[i-1].isalnum(): i -= 1
        while i > 0 and text[i-1].isalnum(): i -= 1
        return i

    def next_word(text, i):
        while i < len(text) and text[i].isalnum(): i += 1
        while i < len(text) and not text[i].isalnum(): i += 1
        return i

    def _wrap(s, w):
        """Split by newlines, then hard-wrap each to width w."""
        lines = []
        for part in s.split('\n'):
            if not part:
                lines.append('')
            else:
                while len(part) > w:
                    lines.append(part[:w])
                    part = part[w:]
                lines.append(part)
        return lines

    def _cursor_visual(text, cursor, w):
        """Convert cursor index to (visual_row, visual_col)."""
        before = text[:cursor]
        lines = _wrap(before, w)
        if not lines: return 0, 0
        return len(lines) - 1, len(lines[-1])

    def draw(buf: ScreenBuffer, inpt, r):
        nonlocal text, cursor
        inner_w = r[2]
        if inner_w < 1: return

        typed = inpt.consume_text()
        if typed:
            text = text[:cursor] + typed + text[cursor:]
            cursor += len(typed)
        # newline: shift+enter (KEY_ALT_ENTER on Windows), '\n', or Ctrl+J
        for i, k in enumerate(inpt._keys):
            if k.name == 'KEY_ALT_ENTER' or str(k) == '\n':
                text = text[:cursor] + '\n' + text[cursor:]
                cursor += 1
                inpt._keys.pop(i)
                break
        if inpt.consume('KEY_LEFT') and cursor > 0: cursor -= 1
        if inpt.consume('KEY_RIGHT') and cursor < len(text): cursor += 1
        if inpt.consume('KEY_BACKSPACE') and cursor > 0:
            text = text[:cursor-1] + text[cursor:]
            cursor -= 1
        if inpt.consume('KEY_DELETE') and cursor < len(text):
            text = text[:cursor] + text[cursor+1:]
        if inpt.consume('KEY_CTRL_BACKSPACE') and cursor > 0:
            new_cursor = prev_word(text, cursor)
            text = text[:new_cursor] + text[cursor:]
            cursor = new_cursor
        if inpt.consume('KEY_CTRL_DELETE') and cursor < len(text):
            text = text[:cursor] + text[next_word(text, cursor):]
        if inpt.consume('KEY_CTRL_LEFT'): cursor = prev_word(text, cursor)
        if inpt.consume('KEY_CTRL_RIGHT'): cursor = next_word(text, cursor)
        if inpt.consume('KEY_ENTER') and text:
            on_submit(text)
            text, cursor = "", 0
            return

        # render multiline with wrapping
        blink = "█" if int(time.time() * 3) % 2 == 0 else " "
        cy, cx = _cursor_visual(text, cursor, inner_w)
        visual_lines = _wrap(text, inner_w)
        if not visual_lines: visual_lines = ['']

        max_visible = r[3]
        scroll = max(0, cy - max_visible + 1)
        for i, line in enumerate(visual_lines[scroll:scroll + max_visible]):
            vi = i + scroll
            if vi == cy:
                buf.puts(r[0], r[1]+i, line[:cx] + blink + line[cx:], txt_color='white')
            else:
                buf.puts(r[0], r[1]+i, line, txt_color='white')

    def get_height(width):
        if width < 1: return 1
        return len(_wrap(text, width)) if text else 1

    draw.get_height = get_height
    return draw



# --- SELECTION MODE UI ---


@overridable
def render_selection_mode_context_name(buf, ctx, x, y):
    selected = ctx is state.current
    approx = "~" if ctx.is_token_count_estimate() else ""
    toks = f" ({approx}{ctx.token_count()//1000}k)"
    spin = "/—\\|"[int(time.time() * 12) % 4]
    suffix = f" {spin}" if ctx.is_running() else ""

    if ctx.is_running(): name_color = 'bright_blue'
    elif ctx.llm_suspended: name_color = 'green'
    else: name_color = 'white'

    buf.puts(x, y, ctx.name, txt_color=name_color, style='bold' if selected else None)
    x += len(ctx.name)
    buf.puts(x, y, toks, txt_color='bright_black')
    x += len(toks)
    buf.puts(x, y, suffix, txt_color='yellow')


@overridable
def render_selection_left(buf, inpt, r):
    x, y, w, h = r
    buf.rect_line(r, txt_color='blue')

    ctxs = sorted(state.contexts.values(), key=lambda c: c.name)
    idx = next((i for i, c in enumerate(ctxs) if c is state.current), 0)

    # navigation
    if inpt.consume('KEY_UP') and idx > 0:
        state.current = ctxs[idx - 1]
    if inpt.consume('KEY_DOWN') and idx < len(ctxs) - 1:
        state.current = ctxs[idx + 1]

    # draw list
    for i, ctx in enumerate(ctxs):
        if i >= h - 2: break
        selected = (ctx is state.current)
        prefix = ">> " if selected else "   "
        row = y + 1 + i
        buf.puts(x + 1, row, prefix, txt_color='red' if selected else None)
        render_selection_mode_context_name(buf, ctx, x + 1 + len(prefix), row)


@overridable
def render_selection_right(buf, r):
    x, y, w, h = r
    buf.rect_line(r, txt_color='blue')

    ctx = state.current
    # header
    buf.puts(x + 2, y + 1, ctx.name, style='bold')
    buf.puts(x + 2 + len(ctx.name) + 2, y + 1, ctx.model, style='dim')

    # token bar
    ratio = ctx.token_count() / ctx.max_tokens if ctx.max_tokens else 0
    bar_w = min(w - 4, 20)
    filled = int(ratio * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)
    buf.puts(x + 2, y + 2, bar, txt_color='cyan')
    approx = "~" if ctx.is_token_count_estimate() else ""
    buf.puts(x + 2 + bar_w + 1, y + 2, f"{approx}{ctx.token_count()//1000}k/{ctx.max_tokens//1000}k", txt_color='cyan')


    # messages
    buf.hline((x + 1, y + 3, w - 2, 1), txt_color='blue')
    row = y + 4
    msgs = ctx.messages or []
    for msg in msgs:
        if row >= y + h - 1: break
        role = msg.role
        content = msg.content if isinstance(msg.content, str) else "<fn>"
        toks = len(content) // 4
        buf.puts(x + 2, row, f"{role} (~{toks//1000}k)", style='dim')
        row += 1
    if not msgs:
        buf.puts(x + 2, row, "(no messages)", style='dim')




def _render_chunks(chunks):
    """Build display string from chunks list."""
    parts = []
    for c in chunks:
        if c.type == "text":
            parts.append(c.content)
        elif c.type == "cot":
            continue
        elif c.type == "tool":
            parts.append(c.content)
    return "".join(parts)

@overridable
def render_work_mode(buf, inpt, r):
    x, y, w, h = r
    ctx = state.current

    # Build per-message output: list of (role, lines)
    message_outputs = []
    for msg in ctx.messages:
        c = _render_chunks(msg.chunks) if msg.role == "assistant" and msg.chunks else msg.get_msg(ctx)
        lines = c.split('\n')
        for renderer in _output_renderers: renderer(msg.role, lines, ctx)
        message_outputs.append((msg.role, lines))
    if ctx.is_running() and not ctx.llm_suspended:
        lines = (_render_chunks(ctx.llm_current_output) + "█").split('\n')
        for renderer in _output_renderers: renderer('assistant', lines, ctx)
        message_outputs.append(('assistant', lines))

    # Draw
    available = h
    scroll_offset = max(0, ctx._prev_height - available)
    row = y - scroll_offset
    for role, lines in message_outputs:
        row += 1  # spacer between messages
        bg = 'bright_black' if role == 'user' else None
        for line in lines:
            if callable(line):
                row += line(buf, x, row, w)
            else:
                drawn = buf.print_wrapped(line, x, row, w, txt_color='white', bg_color=bg)
                if bg:
                    for i in range(drawn):
                        if 0 <= row+i < buf.h:
                            for c in range(w):
                                if buf.bg_colors[row+i][x+c] is None:
                                    buf.bg_colors[row+i][x+c] = bg
                row += drawn
    ctx._prev_height = row - (y - scroll_offset)

@overridable
def render_work_mode_input(buf, inpt, input_r, input_box):
    ctx = state.current
    if ctx.is_running():
        spin = "[" + "/—\\|"[int(time.time() * 12) % 4] + "]"
        elapsed = f"{time.time() - ctx.last_invoke_time_start:.1f}s"
        chunks = ctx.llm_current_output
        x = input_r[0]
        y = input_r[1]
        buf.puts(x, y, spin, txt_color='bright_yellow'); x += 4
        if chunks:
            toks = sum(c.tokens for c in chunks)
            last_type = chunks[-1].type
            label = "thinking..." if last_type == "cot" else "outputting..."
            buf.puts(x, y, label, txt_color='blue'); x += len(label)
            buf.puts(x, y, f" ({toks} toks, {elapsed}) ", txt_color='bright_black')
        else:
            buf.puts(x, y, "invoking...", txt_color='blue'); x += 11
            buf.puts(x, y, f" ({elapsed}) ", txt_color='bright_black')
    else:
        input_box(buf, inpt, input_r)


def _load_plugins():
    plugin_dir = "_ex6"
    if not os.path.isdir(plugin_dir):
        return
    import importlib.util
    import types
    def _register_pkg(name, dir_path):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [os.path.abspath(dir_path)]
            sys.modules[name] = pkg

    _register_pkg("_ex6", plugin_dir)

    for dirpath, dirnames, filenames in os.walk(plugin_dir):
        # skip dirs starting with _  (but not _ex6 root itself)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("_"))
        # register subdirs as subpackages
        rel = os.path.relpath(dirpath, plugin_dir).replace(os.sep, ".")
        pkg_name = "_ex6" if rel == "." else f"_ex6.{rel}"
        _register_pkg(pkg_name, dirpath)
        # load .py files
        for filename in sorted(filenames):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue
            mod_name = f"{pkg_name}.{filename[:-3]}"
            if mod_name in sys.modules:
                continue
            path = os.path.join(dirpath, filename)
            spec = importlib.util.spec_from_file_location(mod_name, path)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)



if __name__ == "__main__":
    _load_plugins()

    state.term = Terminal()
    term = state.term
    buf = ScreenBuffer(term.width, term.height)
    keys = []

    def on_submit(text):
        if text.startswith("/"):
            dispatch_command(text)
        else:
            state.current.invoke(text)
    input_box = make_input(on_submit)

    with term.cbreak(), term.hidden_cursor(), term.fullscreen():
        while True:
            if _fatal_error:
                raise RuntimeError(f"FATAL: {_fatal_error}")
            key = term.inkey(timeout=0.011)
            if key:
                if str(key) == '\x03': break
                if _log_keys:
                    debug_print(f"key: name={key.name!r} str={str(key)!r} code={key.code!r} seq={key.is_sequence}")
                keys.append(key)

            if buf.w != term.width or buf.h != term.height:
                buf = ScreenBuffer(term.width, term.height)

            inpt = InputPass(keys)
            keys = []

            buf.clear()

            # No contexts = show message and block everything
            if not state.contexts:
                msg = "You must create a plugin with Contexts for ex6 to work."
                mx = (term.width - len(msg)) // 2
                my = term.height // 2
                buf.puts(mx, my, msg, txt_color='red')
                buf.flush(term)
                continue

            # Ensure state.current always points to a valid context
            if state.current not in state.contexts.values():
                state.current = next(iter(state.contexts.values()))

            term_r = Region(0,0, term.width, term.height)

            if state.mode == "work":
                # work mode: no boxes, divider, 2 lines bottom padding
                if hasattr(input_box, 'get_height') and not state.current.is_running():
                    input_h = max(1, min(input_box.get_height(term.width), term.height // 2))
                else:
                    input_h = 1
                divider_y = term.height - 4 - input_h - 1
                input_r = Region(0, divider_y + 1, term.width, input_h)
                main_r = Region(0, 0, term.width, divider_y)
            else:
                input_h = 3
                input_r = Region(0, term.height - input_h, term.width, input_h)
                main_r = Region(0, 0, term.width, term.height - input_h)

            if state.mode == "scroll":
                if inpt._keys:  # any key exits scroll mode
                    state.mode = state._prev_mode
                    print(term.enter_fullscreen, end='', flush=True)
            elif state.mode == "work":
                render_work_mode(buf, inpt, main_r)
                div_color = 'bright_yellow' if state.current.is_running() else 'bright_black'
                buf.hline((0, divider_y, term.width, 1), txt_color=div_color)
                render_work_mode_input(buf, inpt, input_r, input_box)
                buf.hline((0, divider_y + 1 + input_h, term.width, 1), txt_color=div_color)
                if inpt.consume('KEY_ESCAPE'):
                    state.mode = "selection"
            elif state.mode == "selection":
                if inpt.consume("KEY_ENTER"):
                    state.mode = "work"
                left, right = main_r.split_horizontal(1, 3)
                render_selection_left(buf, inpt, left)
                render_selection_right(buf, right)
                buf.rect_line(input_r, txt_color='bright_red')
                input_box(buf, inpt, input_r.shrink(1))
            else:
                assert state.mode == "help"
                # press h to toggle help.
                # displays all keybinds for selection-mode
            
            if state.mode != "scroll":
                buf.flush(term)


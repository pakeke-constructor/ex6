
import os
import sys

os.environ.setdefault('ESCDELAY', '25')  # reduce escape key delay (ms)
# (if using SSH, you might want to set this higher. Ask some LLM to explain why.)

sys.modules['ex6'] = sys.modules[__name__]  # so plugins can `import ex6`

from blessed import Terminal
from typing import Union, Tuple, List, Optional, Literal, Callable
import time
from dataclasses import dataclass, field
from typing import Optional
import threading
import inspect
from typing import get_origin, get_args
import copy 
import time
import glob



_commands = {}




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
    current: 'Context' = None  # always valid when contexts is non-empty
    mode: str = "selection"


state = AppState()



@overridable
def invoke_llm(ctx):
    """Override this to use real LLM."""
    for _ in range(60):
        time.sleep(0.1)
        yield ResponseChunk("text", "token ")


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
    tokens: int = 0  # for cot

@dataclass
class LLMResult:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    error: Optional[str] = None


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
    """Ensure tool has (ctx: Context, tool_call_id: str, ...) signature."""
    params = list(inspect.signature(fn).parameters.values())
    if len(params) < 2:
        raise TypeError(f"Tool '{name}' must have at least 2 params: (ctx, tool_call_id)")
    if params[0].annotation not in (Context, 'Context', inspect.Parameter.empty):
        raise TypeError(f"Tool '{name}' first param must be ctx: Context")
    if params[1].annotation not in (str, inspect.Parameter.empty):
        raise TypeError(f"Tool '{name}' second param must be tool_call_id: str")


def tool_to_schema(name: str, fn: Callable) -> dict:
    _validate_tool_sig(name, fn)
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())[2:]  # skip ctx, tool_call_id
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



@dataclass
class Context:
    name: str
    model: str
    messages: list = field(default_factory=list)
    max_tokens: int = 200000
    cost: float = 0.15
    llm_is_running: bool = False
    llm_current_output: list = field(default_factory=list)
    last_invoke_time: float = 0
    llm_result: Optional[LLMResult] = None
    input_stack: list = field(default_factory=list)
    _msg_lock: threading.Lock = field(default_factory=threading.Lock)
    llm_suspended: bool = False

    @property
    def tokens(self) -> int:
        if not self.llm_result: return 0
        return (self.llm_result.input_tokens + self.llm_result.output_tokens) if self.llm_result else 0

    def __post_init__(self):
        state.contexts[self.name] = self

    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other
    def is_running(self): return self.llm_is_running

    def get_tools(self) -> dict[str, Callable]:
        tools = {}
        for m in self.messages:
            tools.update(m.tools)
        return tools

    def get_tool_schemas(self) -> list[dict]:
        return [tool_to_schema(name, fn) for name, fn in self.get_tools().items()]

    def add_tool_result(self, tool_call_id: str, content: str):
        """Add tool result message."""
        with self._msg_lock:
            self.messages.append(Message(role="tool", content=content, tool_call_id=tool_call_id))

    def invoke(self, text, llm_fn=None):
        llm_fn = llm_fn or invoke_llm
        self.messages.append(Message(role="user", content=text))
        self.llm_is_running = True

        def do_llm():
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
            do_llm()
            while self.llm_result and self.llm_result.tool_calls:
                self.llm_suspended = True
                tools = self.get_tools()
                threads = []
                for tc in self.llm_result.tool_calls:
                    fn = tools.get(tc["name"])
                    if fn:
                        t = threading.Thread(target=fn, args=(self, tc["id"]), kwargs=_check_tool_args(fn, tc["args"]))
                        t.start()
                        threads.append(t)
                for t in threads:
                    t.join()
                self.llm_suspended = False
                do_llm()
            self.llm_is_running = False
            self.last_invoke_time = time.time()

        threading.Thread(target=run, daemon=True).start()
    
    def fork(self, new_name: Optional[str] = None) -> 'Context':
        cpy = copy.copy(self)
        cpy.messages = copy.deepcopy(self.messages)
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
        out = term.home
        for y in range(self.h):
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

    def text_contained(self, txt: str, r: Rect, style=None, txt_color=None, bg_color=None, wrap=True, newlines=True) -> int:
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

    def draw(buf, inpt, r):
        nonlocal text, cursor
        typed = inpt.consume_text()
        if typed:
            text = text[:cursor] + typed + text[cursor:]
            cursor += len(typed)
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

        blink = "█" if int(time.time() * 2) % 2 == 0 else " "
        buf.puts(r[0], r[1], "> " + text[:cursor] + blink + text[cursor:], txt_color='white')

    return draw


@overridable
def make_work_input():
    def on_submit(text):
        if text.startswith("/"):
            dispatch_command(text)
        else:
            state.current.invoke(text)
    return make_input(on_submit)


# --- SELECTION MODE UI ---

@overridable
def render_selection_left(buf, inpt, r):
    x, y, w, h = r
    buf.rect_line(r, txt_color='blue')
    buf.puts(x + 2, y, " Contexts ", txt_color='blue')

    ctxs = sorted(state.contexts.values(), key=lambda c: c.name)
    idx = next((i for i, c in enumerate(ctxs) if c is state.current), 0)

    # navigation
    if inpt.consume('KEY_UP') and idx > 0:
        state.current = ctxs[idx - 1]
    if inpt.consume('KEY_DOWN') and idx < len(ctxs) - 1:
        state.current = ctxs[idx + 1]
    if inpt.consume('KEY_ENTER'):
        state.mode = "work"

    # draw list
    now = time.time()
    SPINNER = "/-\\|"
    spin = SPINNER[int(now * 8) % len(SPINNER)]
    for i, ctx in enumerate(ctxs):
        if i >= h - 2: break
        selected = (ctx is state.current)
        prefix = ">> " if selected else "   "
        suffix = f" {spin}" if ctx.is_running() else ""
        toks = f" ({ctx.tokens//1000}k)"

        if ctx.is_running(): txt_color = 'yellow'
        elif now - ctx.last_invoke_time < 360: txt_color = 'white'
        else: txt_color = None

        line = f"{prefix}{ctx.name}{toks}{suffix}"
        buf.puts(
            x + 1, y + 1 + i, line[:w-2],
            style='bold' if selected else ('dim' if not txt_color else None),
            txt_color=txt_color
        )


@overridable
def render_selection_right(buf, r):
    x, y, w, h = r
    buf.rect_line(r, txt_color='blue')
    buf.puts(x + 2, y, " Info ", txt_color='blue')

    ctx = state.current
    # header
    buf.puts(x + 2, y + 1, ctx.name, style='bold')
    buf.puts(x + 2 + len(ctx.name) + 2, y + 1, ctx.model, style='dim')

    # token bar
    ratio = ctx.tokens / ctx.max_tokens if ctx.max_tokens else 0
    bar_w = min(w - 4, 20)
    filled = int(ratio * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)
    buf.puts(x + 2, y + 2, bar, txt_color='cyan')
    buf.puts(x + 2 + bar_w + 1, y + 2, f"{ctx.tokens//1000}k/{ctx.max_tokens//1000}k", style='dim')

    # cost
    buf.puts(x + 2, y + 3, f"${ctx.cost:.2f}", style='dim')

    # messages
    buf.hline((x + 1, y + 4, w - 2, 1), txt_color='blue')
    row = y + 5
    msgs = ctx.messages or []
    for msg in msgs:
        if row >= y + h - 1: break
        role = msg.role
        content = msg.content if isinstance(msg.content, str) else "<fn>"
        toks = len(content) * 4
        buf.puts(x + 2, row, f"{role} ({toks//1000}k)", style='dim')
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
            parts.append(f"[thinking: {c.tokens} tokens]")
        elif c.type == "tool":
            parts.append(c.content)
    return "".join(parts)

@overridable
def render_work_mode(buf, inpt, r):
    x, y, w, h = r
    ctx = state.current
    buf.rect_line(r, txt_color='blue')
    buf.puts(x + 2, y, f" {ctx.name} ", txt_color='blue')

    # Build lines from messages (most recent that fit)
    lines = []
    for msg in ctx.messages:
        role = msg.role
        # Use chunks for assistant msgs if available, else fallback to content
        if role == "assistant" and msg.chunks:
            content = _render_chunks(msg.chunks)
        else:
            content = msg.get_msg(ctx)
        if role == "user": txt_color, s = "cyan", None
        elif role == "assistant": txt_color, s = "white", None
        else: txt_color, s = None, "dim"
        lines.append((content, s, txt_color))
    if ctx.is_running():
        content = _render_chunks(ctx.llm_current_output) + "█"
        lines.append((content, None, "yellow"))

    # Render from top down, showing recent
    row = y + 1
    for content, s, txt_color in lines:
        rows_used = buf.text_contained(content, (x+1, row, w-2, h-2-row+y), style=s, txt_color=txt_color)
        row += rows_used
        if row >= y + h - 1: break


def _load_plugins():
    plugin_dir = "_ex6"
    if not os.path.isdir(plugin_dir):
        return
    for path in glob.glob(os.path.join(plugin_dir, "*.py")):
        with open(path, "r", encoding="utf-8") as f:
            exec(compile(f.read(), path, "exec"), {"__name__": "__plugin__", "__file__": path})




if __name__ == "__main__":
    _load_plugins()

    term = Terminal()
    buf = ScreenBuffer(term.width, term.height)
    keys = []
    selection_input = make_input(lambda t: None)
    work_input = make_work_input()

    with term.cbreak(), term.hidden_cursor(), term.fullscreen():
        while True:
            key = term.inkey(timeout=0.011)
            if key:
                if str(key) == '\x03': break
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

            main_r = Region(0, 0, term.width, term.height - 1)
            input_r = Region(0, term.height - 1, term.width, 1)

            if state.mode == "selection":
                left, right = main_r.split_horizontal(1, 2)
                render_selection_left(buf, inpt, left)
                render_selection_right(buf, right)
                selection_input(buf, inpt, input_r)
            else:  # work mode
                if inpt.consume('KEY_ESCAPE'):
                    state.mode = "selection"
                else:
                    render_work_mode(buf, inpt, main_r)
                    work_input(buf, inpt, input_r)

            buf.flush(term)


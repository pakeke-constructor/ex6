
import os

os.environ.setdefault('ESCDELAY', '25')  # reduce escape key delay (ms)
# (if using SSH, you might want to set this higher. Ask some LLM to explain why.)


from blessed import Terminal
from typing import Tuple
import time
from region import Region
from dataclasses import dataclass, field
from typing import Optional
import threading
import copy 
import time




Rect = Tuple[int, int, int, int]  # (x, y, w, h)

SPINNER = "/-\\|"

class ScreenBuffer:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.chars = [[' '] * w for _ in range(h)]
        self.styles = [[None] * w for _ in range(h)]

    def put(self, x, y, char, style=None):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.chars[y][x] = char
            self.styles[y][x] = style

    def puts(self, x, y, text, style=None):
        for i, c in enumerate(text):
            self.put(x + i, y, c, style)

    def clear(self):
        for row in self.chars: row[:] = [' '] * self.w
        for row in self.styles: row[:] = [None] * self.w

    def flush(self, term):
        out = term.home
        for y in range(self.h):
            for x in range(self.w):
                c, s = self.chars[y][x], self.styles[y][x]
                styled = getattr(term, s, None) if s else None
                out += styled(c) if styled else c
        print(out, end='', flush=True)

    def fill(self, r: Rect, char='█', style=None):
        x, y, w, h = r
        for row in range(y, y + h):
            for col in range(x, x + w):
                self.put(col, row, char, style)

    def rect_line(self, r: Rect, style=None):
        x, y, w, h = r
        if w < 2 or h < 2: return
        for col in range(x + 1, x + w - 1):
            self.put(col, y, '─', style)
            self.put(col, y + h - 1, '─', style)
        for row in range(y + 1, y + h - 1):
            self.put(x, row, '│', style)
            self.put(x + w - 1, row, '│', style)
        self.put(x, y, '┌', style)
        self.put(x + w - 1, y, '┐', style)
        self.put(x, y + h - 1, '└', style)
        self.put(x + w - 1, y + h - 1, '┘', style)

    def hline(self, r: Rect, style=None):
        x, y, w, _ = r
        for i in range(w):
            self.put(x + i, y, '─', style)

    def vline(self, r: Rect, style=None):
        x, y, _, h = r
        for i in range(h):
            self.put(x, y + i, '│', style)

    def text_contained(self, txt: str, r: Rect, style=None, wrap=True, newlines=True) -> int:
        x, y, w, h = r
        if not newlines:
            txt = txt.replace('\n', ' ')
        txt = txt.replace('\r\n', '\n').replace('\r', '\n')
        row, col = 0, 0
        for c in txt:
            if c == '\n': row += 1; col = 0; continue
            if wrap and col >= w: row += 1; col = 0
            if row >= h or (not wrap and col >= w): continue
            self.put(x + col, y + row, c, style)
            col += 1
        return row + 1 if col > 0 or row == 0 else row




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
    contexts: set = field(default_factory=set)
    current: Optional['Context'] = None
    mode: str = "selection"


state = AppState()



@overridable
def invoke_llm(ctx):
    """Override this to use real LLM."""
    for _ in range(60):
        time.sleep(0.1)
        yield "token "


@dataclass
class Context:
    name: str
    model: str = "opus-4.5"
    messages: list = field(default_factory=list)
    tokens: int = 32000
    max_tokens: int = 200000
    cost: float = 0.15
    llm_running: bool = False
    llm_output: str = ""
    last_llm_time: float = 0
    messages: list = field(default_factory=list)
    input_stack: list = field(default_factory=list)

    def __post_init__(self):
        def on_submit(t):
            if t.startswith("/"): dispatch_command(t)
            else: self.invoke(t)
        self.input_stack = [make_input(on_submit)]
        state.contexts.add(self)

    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other

    def invoke(self, text, llm_fn=None):
        llm_fn = llm_fn or invoke_llm
        self.messages.append({"role": "user", "content": text})
        self.llm_running = True
        self.llm_output = ""
        def run():
            for token in llm_fn(self):
                self.llm_output += token
            self.messages.append({"role": "assistant", "content": self.llm_output})
            self.llm_running = False
            self.last_llm_time = time.time()
        threading.Thread(target=run, daemon=True).start()
    
    def fork(self) -> Context:
        cpy = copy.copy(self)
        cpy.messages = copy.deepcopy(self.messages)
        cpy.input_stack = []
        cpy.__post_init__()  # fresh input handlers
        return cpy

    def push_ui(self, draw_fn):
        self.input_stack.append(draw_fn)




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
        buf.puts(r[0], r[1], "> " + text[:cursor] + blink + text[cursor:], 'white')

    return draw

def make_work_input():
    def on_submit(text):
        if state.current:
            state.current.invoke(text)
    return make_input(on_submit)


# --- SELECTION MODE UI ---

def render_selection_left(buf, inpt, r):
    x, y, w, h = r
    buf.rect_line(r, 'blue')
    buf.puts(x + 2, y, " Contexts ", 'blue')

    ctxs = sorted(state.contexts, key=lambda c: c.name)
    if not ctxs:
        buf.puts(x + 2, y + 1, "(no contexts)", 'dim')
        return

    idx = next((i for i, c in enumerate(ctxs) if c is state.current), 0)

    # navigation
    if inpt.consume('KEY_UP') and idx > 0:
        state.current = ctxs[idx - 1]
    if inpt.consume('KEY_DOWN') and idx < len(ctxs) - 1:
        state.current = ctxs[idx + 1]
    if inpt.consume('KEY_ENTER') and state.current:
        state.mode = "work"

    # draw list
    now = time.time()
    spin = SPINNER[int(now * 8) % len(SPINNER)]
    for i, ctx in enumerate(ctxs):
        if i >= h - 2: break
        selected = (ctx is state.current)
        prefix = ">> " if selected else "   "
        suffix = f" {spin}" if ctx.llm_running else ""
        toks = f" ({ctx.tokens//1000}k)"

        if ctx.llm_running: style = 'yellow'
        elif now - ctx.last_llm_time < 360: style = 'white'
        else: style = 'dim'

        line = f"{prefix}{ctx.name}{toks}{suffix}"
        buf.puts(x + 1, y + 1 + i, line[:w-2], 'bold' if selected else style)


def render_selection_right(buf, r):
    x, y, w, h = r
    buf.rect_line(r, 'blue')
    buf.puts(x + 2, y, " Info ", 'blue')

    ctx = state.current
    if not ctx:
        buf.puts(x + 2, y + 1, "(no context selected)", 'dim')
        return

    # header
    buf.puts(x + 2, y + 1, ctx.name, 'bold')
    buf.puts(x + 2 + len(ctx.name) + 2, y + 1, ctx.model, 'dim')

    # token bar
    ratio = ctx.tokens / ctx.max_tokens if ctx.max_tokens else 0
    bar_w = min(w - 4, 20)
    filled = int(ratio * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)
    buf.puts(x + 2, y + 2, bar, 'cyan')
    buf.puts(x + 2 + bar_w + 1, y + 2, f"{ctx.tokens//1000}k/{ctx.max_tokens//1000}k", 'dim')

    # cost
    buf.puts(x + 2, y + 3, f"${ctx.cost:.2f}", 'dim')

    # messages
    buf.hline((x + 1, y + 4, w - 2, 1), 'blue')
    row = y + 5
    msgs = ctx.messages or []
    for msg in msgs:
        if row >= y + h - 1: break
        role = msg.get("name", msg.get("role", "?"))
        toks = len(msg.get("content", "")) * 4
        buf.puts(x + 2, row, f"{role} ({toks})", 'dim')
        row += 1
    if not msgs:
        buf.puts(x + 2, row, "(no messages)", 'dim')


# --- WORK MODE UI ---

def render_work_mode(buf, inpt, r):
    x, y, w, h = r
    ctx = state.current
    assert ctx
    buf.rect_line(r, 'blue')
    buf.puts(x + 2, y, f" {ctx.name} ", 'blue')

    # Build lines from messages (most recent that fit)
    lines = []
    for msg in ctx.messages:
        role, content = msg["role"], msg.get("content", "")
        style = "cyan" if role == "user" else ("white" if role == "assistant" else "dim")
        lines.append((content, style))
    if ctx.llm_running:
        lines.append((ctx.llm_output + "█", "yellow"))

    # Render from top down, showing recent
    row = y + 1
    for content, style in lines:
        rows_used = buf.text_contained(content, (x+1, row, w-2, h-2-row+y), style)
        row += rows_used
        if row >= y + h - 1: break


# --- MAIN ---

if __name__ == "__main__":
    # dummy contexts
    c1 = Context("ctx1", messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ])
    c2 = Context("ctx2", model="sonnet-4", tokens=5000)
    c3 = Context("foobar", tokens=45000, cost=0.08)
    state.contexts = {c1, c2, c3}
    state.current = c1

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



import os
import sys
import json
import hashlib
from pathlib import Path
from collections import deque, Counter
from datetime import datetime
import math
import itertools
import functools
import re


os.environ.setdefault('ESCDELAY', '25')  # reduce escape key delay (ms)
# (if using SSH, you might want to set this higher. Ask some LLM to explain why.)

sys.modules['ex6'] = sys.modules[__name__]  # so plugins can `import ex6`

class _StdoutSink:
    def __init__(self, debug):
        self.debug = debug

    def write(self, s):
        s = s.strip()
        if s:
            self.debug(f"[stdout] {s}")

    def flush(self): pass
    def isatty(self): return False

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from blessed import Terminal
from typing import Union, Tuple, List, Optional, Literal, Callable
import time
from dataclasses import dataclass, field, InitVar
from typing import Optional,Any
import threading
import inspect
from typing import get_origin, get_args, get_type_hints
import copy
import time
import glob
import difflib
import atexit
import shutil
import tempfile
from contextlib import closing


# 6 lines per scroll
ARROW_SCROLL_SPEED: int = 6



ESC_DELAY: float = 0
# NOTE: if you are running ex6 through ssh or something,
# (eg on a VPS) you might want to set this to 25 or 50 or something.
# To understand why, ask an LLM about it


# Type aliases for output rendering
Color = Union[str, Tuple[int, int, int]]
RenderFn = Callable[['ScreenBuffer', int, int, int], int]  # fn(buf, x, y, w) -> rows
OutputLine = Union[str, RenderFn]  # str or render fn
OutputRendererFn = Callable[[list, 'Message', 'Context'], None]  # fn(lines, msg, ctx) -> None

def _declaration(kind):
    def mark(fn):
        setattr(fn, "_ex6_declaration", kind)
        return fn
    return mark


CONTEXT_HOOK_KINDS = ("after_tool_calls", "after_llm_turn", "output_renderer", "context_removed")

after_tool_calls = _declaration("after_tool_calls")
context_removed = _declaration("context_removed")
after_llm_turn = _declaration("after_llm_turn")
output_renderer = _declaration("output_renderer")
command = _declaration("command")
_overridable = _declaration("overridable")
override = _declaration("override")
_handleable = _declaration("handleable")


def handler(fn=None, *, order=0):
    def mark(value):
        setattr(value, "_ex6_declaration", "handler")
        setattr(value, "_ex6_order", order)
        return value
    return mark if fn is None else mark(fn)


def key(*seqs):
    """Bind fn(tui, count: int) to selection-mode key sequences, eg @ex6.key("dd", "<C-x>")."""
    def mark(fn):
        fn._ex6_declaration = "key"
        fn._ex6_keys = seqs
        return fn
    return mark


def split_keys(seq: str) -> tuple:
    return tuple(re.findall(r"<[^>]+>|.", seq))


def key_token(k) -> str:
    """Blessed keystroke -> keymap token: 'a', ' ', '<CR>', '<Esc>', '<C-x>', '<UP>'."""
    s = str(k)
    if k.name == 'KEY_ENTER' or s in ('\r', '\n'): return '<CR>'
    if k.name == 'KEY_ESCAPE' or s == '\x1b': return '<Esc>'
    if s == '\t': return '<Tab>'
    if len(s) == 1 and ord(s) < 32: return f"<C-{chr(ord(s) + 64).lower()}>"
    if k.is_sequence and k.name: return f"<{k.name.removeprefix('KEY_')}>"
    return s





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

@command
def dbg(tui):
    tui.enter_scroll_mode()
    print("="*30)
    print("DEBUG LOG")
    print("="*30)
    print(f"({len(tui.app.debug_buffer)} entries)\n")
    for line in tui.app.debug_buffer:
        print(line)

@command
def yolo(tui):
    'Toggle auto-approve tools.'
    ctx = tui.current
    if not ctx: return
    ctx.yolo = not ctx.yolo


def get_folder(owner: Optional[str] = None) -> Path:
    """Returns app data folder namespaced by owner (plugin id): %APPDATA%/ex6/<owner> on Windows, ~/ex6/<owner> on Unix."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path.home()
    if owner:
        folder = base / "ex6" / owner
    else:
        folder = base / "ex6"
    folder.mkdir(parents=True, exist_ok=True)
    return folder




class DailyBudget:
    def __init__(self, limit: Optional[float] = 15.0):
        self.limit = limit
        self.cost = 0.0
        self.date = ""
        self.usage_mtime = None
        self.lock = threading.Lock()

    def get_limit(self) -> float:
        return 1000 if self.limit is None else self.limit

    def _ensure_loaded(self):
        today = datetime.now().strftime("%Y-%m-%d")
        path = get_folder() / "usage.json"
        try:
            mtime = path.stat().st_mtime
        except:
            mtime = None
        if self.date == today and mtime == self.usage_mtime:
            return
        self.cost = 0.0
        self.date = today
        try:
            data = json.loads(path.read_text())
            if data.get("date") == today:
                self.cost = data.get("cost", 0.0)
        except: pass
        self.usage_mtime = mtime

    def get_cost(self) -> float:
        with self.lock:
            self._ensure_loaded()
            return self.cost

    def is_over(self) -> bool:
        return self.get_cost() >= self.get_limit()

    def add_cost(self, cost: float):
        with self.lock:
            self._ensure_loaded()
            self.cost += cost
            path = get_folder() / "usage.json"
            tmp = path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps({"date": self.date, "cost": self.cost}))
                os.replace(tmp, path)
                self.usage_mtime = path.stat().st_mtime
            except: pass


_tiktoken_enc = None
_o200k_cache = {}  # hash(s) -> count; selection mode re-counts every message every frame

def _o200k(s: str) -> int:
    global _tiktoken_enc
    key = hash(s)
    if key in _o200k_cache:
        return _o200k_cache[key]
    if _tiktoken_enc is None:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("o200k_base")
    if len(_o200k_cache) > 100_000:
        _o200k_cache.clear()
    n = _o200k_cache[key] = len(_tiktoken_enc.encode(s, disallowed_special=()))
    return n


def get_token_estimate(s: str, model: str = "") -> int:
    if model.startswith("openai/"):
        return _o200k(s)
    elif model.startswith("anthropic/"):
        # Claude 5.5: real ~= 1.71 * o200k, fit from invokes.db input deltas.
        return math.ceil(_o200k(s) * 1.7)
    # Other: mixed prose/code/JSON estimate with safety margin.
    return math.ceil(len(s.encode("utf-8")) / 3.2)


MAX_TOOL_OUTPUT_CHARACTERS = 150000


@dataclass(frozen=True)
class Attachment:
    path: str
    mime_type: str


@dataclass(frozen=True)
class ImageAttachment(Attachment):
    width: int
    height: int
    detail: Literal["auto", "low", "high"] = "auto"


@dataclass(frozen=True)
class AudioAttachment(Attachment):
    pass


@dataclass(frozen=True)
class InputValue:
    text: str
    attachments: dict[str, Attachment] = field(default_factory=dict)


@dataclass
class ToolResult:
    text: str
    attachments: tuple[Attachment, ...] = ()


@dataclass
class PendingTool:
    call: dict
    result: ToolResult | None = None


@dataclass
class Operation:
    """Tool-owned work. poll(wait) returns None while pending, str/ToolResult
    after cleanup; poll(0) must not block. Core calls cancel at most once,
    between polls. Plugins own deadlines and cleanup, including on failure."""
    cancel: Callable[[], None]
    poll: Callable[[float], str | ToolResult | None]


def tool_result_text(value) -> str:
    if isinstance(value, ToolResult):
        return value.text
    return str(value or "")


_attachment_dir = tempfile.mkdtemp(prefix="ex6-attachments-")
atexit.register(shutil.rmtree, _attachment_dir, ignore_errors=True)


def store_attachment(data: bytes, suffix: str) -> str:
    digest = hashlib.sha256(data).hexdigest()
    path = os.path.join(_attachment_dir, digest + suffix)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


# When True, the TUI polls the clipboard every few frames and collapses
# consecutive spaces copied from the fixed-width ScreenBuffer layout.
# Set to False in a plugin if you don't want this behaviour.
CLIPBOARD_CLEANING = True
_clipboard_last_seen = ""  # last-seen clipboard text for fast skip

def _is_probably_equal(a: str, b: str):
    """O(1) heuristic: length + 10 sampled chars at stepped offsets."""
    n = len(a)
    if n != len(b): return False
    step = max(1, n // 10)
    for i in range(0, n, step):
        if a[i] != b[i]: return False
    return True

def _try_clean_clipboard():
    """
    Reads clipboard text and collapses consecutive spaces outside leading
    indentation. Uses _is_probably_equal() to skip work when the clipboard
    hasn't changed (O(1) even for huge strings).
    """
    global _clipboard_last_seen
    import re
    try:
        import copykitten
        raw = copykitten.paste()
    except Exception:
        return
    if _is_probably_equal(raw, _clipboard_last_seen):
        return
    _clipboard_last_seen = raw
    parts = re.split(r'(\r\n|\r|\n)', raw)
    for i in range(0, len(parts), 2):
        line = parts[i]
        indentation = len(line) - len(line.lstrip(' \t'))
        parts[i] = line[:indentation] + re.sub(r' {2,}', ' ', line[indentation:])
    cleaned_text = ''.join(parts)
    if cleaned_text != raw:
        try:
            copykitten.copy(cleaned_text)
            _clipboard_last_seen = cleaned_text
        except Exception:
            pass


def clipboard_image_attachment() -> ImageAttachment | None:
    import io
    from PIL import Image, ImageGrab

    try:
        value = ImageGrab.grabclipboard()
    except (NotImplementedError, OSError):
        return None
    if isinstance(value, list):
        image = None
        for path in value:
            try:
                image = Image.open(path)
                break
            except (OSError, ValueError):
                pass
        if image is None:
            return None
    elif isinstance(value, Image.Image):
        image = value
    else:
        return None

    opened_file = isinstance(value, list)
    original = image
    try:
        width, height = image.size
        if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
            image = image.convert("RGBA")
        output = io.BytesIO()
        image.save(output, format="PNG")
    finally:
        if opened_file:
            original.close()

    path = store_attachment(output.getvalue(), ".png")
    return ImageAttachment(path, "image/png", width, height)


@dataclass
class Theme:
    name: str = "default"
    text: Color = "white"
    muted: Color = "bright_black"
    text_alt: Color = (200, 150, 100)  # soft orange: secondary text that should read without looking like an error
    cot: Color = "red"
    user_background: Color = "black"
    accent: Color = "blue"
    accent_alt: Color = "cyan"
    success: Color = "green"
    warning: Color = "yellow"
    error: Color = "red"
    running: Color = "bright_blue"
    invoking: Color = "bright_yellow"
    selection: Color = "red"
    error_bg: Color = (80, 0, 0)
    diff_add_bg: Color = (18, 60, 18)
    diff_del_bg: Color = (60, 18, 18)
    md_bullet: Color = "cyan"
    md_code: Color = "green"
    md_link: Color = "blue"
    md_italic: Color = "magenta"
    md_bold: Color = "bright_white"



class App:
    def __init__(self):
        self.commands = {}
        self.keys = {}  # token tuple (see split_keys) -> fn(tui, count)
        self.context_hooks = {kind: [] for kind in CONTEXT_HOOK_KINDS}
        self.overrides = {}
        self.default_implementations = {}
        self._overridden = set()
        self.handleables = {}
        self.handlers = {}
        self.theme = Theme()
        self.context_schemas = {}
        self.plugin_data = {}
        self.diagnostics = {}
        self.budget = DailyBudget()
        self.debug_buffer = deque(maxlen=1000)
        self.fatal_error = None
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.tui: Optional[TUI] = None
        self.long_horizon = False # True when no human is watching; (tools must not block for user input)
        self._registered_modules = set()
        self._setup_modules = set()
        self._register_module(sys.modules[__name__])
        self.handlers["approve_tool"].append((100, _approve_yolo))

    def debug_print(self, *args, **kwargs):
        ts = datetime.now().strftime("%H:%M:%S")
        msg = " ".join(str(a) for a in args)
        self.debug_buffer.append(f"[{ts}] {msg}")

    def command(self, fn):
        params = list(inspect.signature(fn).parameters.values())[1:]
        empty = inspect.Parameter.empty
        spec = [(p.name, str if p.annotation is empty else p.annotation, None if p.default is empty else p.default) for p in params]
        self.commands[fn.__name__.lower()] = (fn, spec)
        return fn

    def bind(self, seq, fn):
        """Bind a key sequence. Raises if it's a prefix of (or prefixed by) another binding: there are no timeouts."""
        toks = split_keys(seq)
        for other in self.keys:
            n = min(len(other), len(toks))
            if other[:n] == toks[:n]:
                raise RuntimeError(f"key {seq!r} conflicts with {''.join(other)!r}")
        self.keys[toks] = fn
        return fn

    def add_context_hook(self, kind, fn):
        self.context_hooks[kind].append(fn)
        return fn

    def run_context_hooks(self, kind, ctx, *args):
        for fn in self.context_hooks[kind]:
            fn(*args, ctx)
        for fn in ctx.context_hooks[kind]:
            fn(*args, ctx)

    def output_renderer(self, fn):
        return self.add_context_hook("output_renderer", fn)

    def after_tool_calls(self, fn):
        return self.add_context_hook("after_tool_calls", fn)

    def after_llm_turn(self, fn):
        return self.add_context_hook("after_llm_turn", fn)

    def _register_overridable(self, fn):
        self.default_implementations[fn.__name__] = fn
        self.overrides.setdefault(fn.__name__, fn)
        return fn

    def override(self, fn):
        name = fn.__name__
        if name not in self.overrides:
            raise RuntimeError(f"'{name}' not overridable")
        if name in self._overridden:
            raise RuntimeError(f"'{name}' already overridden")
        self._overridden.add(name)
        self.overrides[name] = fn
        return fn

    def _register_handleable(self, fn):
        self.handleables[fn.__name__] = fn
        self.handlers.setdefault(fn.__name__, [])
        return fn

    def handler(self, fn):
        name = fn.__name__
        if name not in self.handleables:
            raise RuntimeError(f"'{name}' not handleable")
        self.handlers[name].append((getattr(fn, "_ex6_order", 0), fn))
        self.handlers[name].sort(key=lambda item: item[0])
        return fn

    def get_implementation(self, name, default=False):
        registry = self.default_implementations if default else self.overrides
        return registry[name]

    def call(self, name, *args, **kwargs):
        """Call handlers until one returns truthy, otherwise fall back to the default implementation."""
        if name in self.handleables:
            for _, fn in self.handlers[name]:
                result = fn(*args, **kwargs)
                if result:
                    return result
            return self.handleables[name](*args, **kwargs)
        return self.overrides[name](*args, **kwargs)

    def pipeline(self, name, value, *args, **kwargs):
        """Chain of transforms: each handler can modify the value, then the default gets the final result."""
        for _, fn in self.handlers[name]:
            result = fn(value, *args, **kwargs)
            if result is not None:
                value = result
        return self.handleables[name](value, *args, **kwargs)

    def iter_commands(self):
        return self.commands.items()

    def dispatch_command(self, tui: TUI, text: str):
        if not text.startswith("/"): return False
        body = text[1:].strip()
        if not body: return False
        name, sep, rest = body.partition(" ")
        name = name.lower()
        if name not in self.commands: return True
        fn, spec = self.commands[name]
        args = ([rest.strip()] if sep else []) if len(spec) == 1 else (rest.split() if sep else [])
        parsed = [_coerce_arg(args[i], typ) if i < len(args) else default for i, (_, typ, default) in enumerate(spec)]
        return fn(tui, *parsed)

    def _register_module(self, module):
        if module.__name__ in self._registered_modules:
            return
        self._registered_modules.add(module.__name__)
        for value in vars(module).values():
            if not callable(value) or getattr(value, "__module__", None) != module.__name__:
                continue
            kind = getattr(value, "_ex6_declaration", None)
            if kind in CONTEXT_HOOK_KINDS: self.add_context_hook(kind, value)
            elif kind == "command": self.command(value)
            elif kind == "key":
                for seq in value._ex6_keys: self.bind(seq, value)
            elif kind == "overridable": self._register_overridable(value)
            elif kind == "override": self.override(value)
            elif kind == "handleable": self._register_handleable(value)
            elif kind == "handler": self.handler(value)

    def setup_module(self, module):
        if module.__name__ in self._setup_modules:
            return
        self._setup_modules.add(module.__name__)
        setup = getattr(module, "setup", None)
        if setup:
            setup(self)

    def load_context(self, serialized_context):
        payload = json.loads(serialized_context)
        schema_id = payload.get("schema_id")
        if schema_id not in self.context_schemas:
            raise ValueError(f"Unknown context schema: {schema_id!r}")
        return self.context_schemas[schema_id].clone_with_context(serialized_context)


def run_tui():
    app = App()
    tui = TUI(app)
    app.tui = tui
    real_stdout, real_stderr = sys.stdout, sys.stderr
    app.stdout, app.stderr = real_stdout, real_stderr
    old_thread_hook = threading.excepthook

    def thread_excepthook(args):
        import traceback
        tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        app.debug_print(f"THREAD CRASH [{args.thread}]:\n{tb}")
        app.fatal_error = f"{args.exc_type.__name__}: {args.exc_value}"

    threading.excepthook = thread_excepthook
    try:
        _load_plugins(app)
        if not tui.contexts:
            if input("You haven't defined any contexts. Initiate auto setup? [y/N] ").strip().lower() not in ("y", "yes"):
                return
            bootstrap()
            _load_plugins(app)
        _run_tui_loop(tui, real_stdout, real_stderr)
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        threading.excepthook = old_thread_hook
        app.tui = None


@_handleable
def invoke_llm(ctx):
    """Provider handlers return the generator fn that runs ctx, or None to abstain."""
    raise RuntimeError(f"no provider handles model {ctx.model!r}")


@_handleable
def supports_audio(ctx):
    """Provider handlers return True if ctx's model accepts audio input."""
    return False


class NativeTool:
    """Provider-hosted tool (eg web search). Goes in tool lists like a normal tool, but the
    provider runs it: no schema is sent, ex6 never calls it, approve_tool is not consulted.
    Providers raise on names they don't support. Calls land in Message.native_calls."""
    def __init__(self, name: str):
        self.__name__ = name

    def __repr__(self):
        return f"NativeTool({self.__name__!r})"


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, ToolResult, Callable[['Context'], str]]
    tools: list[Callable] = field(default_factory=list)
    tool_calls: Optional[list] = None  # for assistant msgs with tool calls
    tool_call_id: Optional[str] = None  # for tool result msgs
    native_calls: Optional[list] = None  # provider-run NativeTool calls ({id, name, args, result}); display/debug only, never sent back
    overview: Optional[str] = None  # short label for display (e.g. in selection panel)
    attachments: dict[str, Attachment] = field(default_factory=dict)
    _snapshot: Optional[str] = field(default=None, repr=False) # Snapshot ensures caching holds when we have callable messages with dynamic content
    _current_tokens: Optional[int] = field(default=None, repr=False)
    _render_cache: Optional[tuple] = field(default=None, repr=False, compare=False)  # (text, theme, lines): output_renderer results

    def __post_init__(self):
        for fn in self.tools:
            _validate_tool_sig(fn.__name__, fn)

    def with_tools(self, tools: list[Callable]):
        assert self.role == "system"
        for fn in tools:
            _validate_tool_sig(fn.__name__, fn)
        by_name = {fn.__name__: fn for fn in self.tools}
        by_name.update({fn.__name__: fn for fn in tools}) # same name replaces in place (keeps schema order)
        clone = copy.copy(self)
        clone.tools = list(by_name.values())
        return clone

    def without_tools(self, *tools: Union[str, Callable]):
        assert self.role == "system"
        names = {t if isinstance(t, str) else t.__name__ for t in tools}
        missing = names - {fn.__name__ for fn in self.tools}
        assert not missing, f"tools not present: {missing}"
        clone = copy.copy(self)
        clone.tools = [fn for fn in self.tools if fn.__name__ not in names]
        return clone

    def get_msg(self, ctx: 'Context'):
        c = self.content
        if self._snapshot:
            return self._snapshot
        if callable(c):
            self._snapshot = c(ctx)
            return self._snapshot
        return c


@dataclass
class ResponseChunk:
    type: Literal["text", "cot"]
    content: str = ""


@dataclass
class LLMResult:
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list = field(default_factory=list)
    finish_reason: str = "stop"
    error: Optional[str] = None
    cost: Optional[float] = None
    native_calls: list = field(default_factory=list)  # see Message.native_calls
    cached_tokens: int = 0  # input tokens read from prompt cache (part of input_tokens)
    cache_write_tokens: int = 0  # input tokens written to prompt cache (part of input_tokens)
    raw_usage: dict = field(default_factory=dict)  # provider-native usage payload; debug/tests only


@dataclass
class ToolCall:
    """A single tool-row for display. call: {id, name, args}. status: running | ok | error."""
    call: dict
    result: Any = None
    status: str = "running"
    tokens: int = 0



def _resolve_type_hint(h):
    if not isinstance(h, str):
        return h
    return {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
    }.get(h, h)


def _type_hint_name(h):
    return h.__name__ if hasattr(h, '__name__') else str(h)


def _strong_isinstance(obj: object, type_hint) -> bool:
    type_hint = _resolve_type_hint(type_hint)
    if type_hint is Any:
        return True
    if isinstance(type_hint, str):
        hint = type_hint.rsplit('.', 1)[-1]
        return type(obj).__name__ == hint

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is None:
        return isinstance(obj, type_hint)

    if origin is Union:
        return any(_strong_isinstance(obj, arg) for arg in args)

    if not isinstance(obj, origin):
        return False

    if origin is list:
        assert isinstance(obj, list)
        if not args:
            return True
        (item_type,) = args
        return all(_strong_isinstance(item, item_type) for item in obj)

    if origin is tuple:
        assert isinstance(obj, tuple)
        if not args:
            return True
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_strong_isinstance(item, args[0]) for item in obj)
        if len(obj) != len(args):
            return False
        return all(_strong_isinstance(item, item_type) for item, item_type in zip(obj, args))

    return False


def _coerce_to_type_hint(val, type_hint):
    type_hint = _resolve_type_hint(type_hint)

    if type_hint is Any or _strong_isinstance(val, type_hint):
        return val, True

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is Union:
        for arg in args:
            coerced, ok = _coerce_to_type_hint(val, arg)
            if ok:
                return coerced, True
        return val, False

    if type_hint in (int, float) and isinstance(val, str):
        try:
            return type_hint(val), True
        except Exception:
            return val, False

    if (origin is list or type_hint is list) and isinstance(val, tuple):
        val = list(val)
        return val, _strong_isinstance(val, type_hint)

    if origin is not tuple and type_hint is not tuple:
        return val, False
    if isinstance(val, list):
        val = tuple(val)
    if not isinstance(val, tuple):
        return val, False
    if not val or not args:
        return val, True

    item_hints = args
    if len(args) == 2 and args[1] is Ellipsis:
        item_hints = (args[0],) * len(val)
    if len(val) != len(item_hints):
        return val, False

    out = []
    for item, item_hint in zip(val, item_hints):
        coerced, ok = _coerce_to_type_hint(item, item_hint)
        if not ok:
            return val, False
        out.append(coerced)
    return tuple(out), True


def _check_tool_args(fn: Callable, args: dict) -> dict:
    """Validate args against fn annotations, raise TypeError if mismatch."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = fn.__annotations__

    for name, val in args.items():
        if name not in hints:
            continue
        hint = hints[name]
        coerced, ok = _coerce_to_type_hint(val, hint)
        if not ok:
            raise TypeError(f"{fn.__name__}: arg '{name}' expected {_type_hint_name(hint)}, got {type(val).__name__}")
        args[name] = coerced
    return args


_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", tuple: "array", dict: "object"}


def _type_to_schema(type_hint):
    type_hint = _resolve_type_hint(type_hint)
    origin = get_origin(type_hint)

    if origin is Union:
        non_none = [a for a in get_args(type_hint) if a is not type(None)]
        type_hint = non_none[0] if non_none else str
        return _type_to_schema(type_hint)

    if origin in (list, tuple):
        args = get_args(type_hint)
        item_type = args[0] if args else str
        return {"type": "array", "items": _type_to_schema(item_type)}

    if origin is dict:
        return {"type": "object"}

    if type_hint in (list, tuple):
        return {"type": "array", "items": {"type": "string"}}

    return {"type": _TYPE_MAP.get(type_hint, "string")}


def _validate_tool_sig(name: str, fn: Callable):
    """Ensure tool has (ctx: Context, ...) signature."""
    if isinstance(fn, NativeTool):
        return
    params = list(inspect.signature(fn).parameters.values())
    if not params:
        raise TypeError(f"Tool '{name}' first param must be ctx: Context")
    ann = params[0].annotation
    if ann is inspect.Parameter.empty:
        return
    try:
        ann = get_type_hints(fn).get(params[0].name, ann)
    except Exception:
        pass
    if ann is not Context:
        raise TypeError(f"Tool '{name}' first param must be ctx: Context, got {ann!r}")


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
        props[pname] = _type_to_schema(ptype)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (fn.__doc__ or "").strip(),
            "parameters": {"type": "object", "properties": props, "required": required}
        }
    }



SPIN = "/-\\|"

@_handleable
def render_tool_call(ctx, call, result):
    """Handlers return spans [(text, theme_color_key), ...] for the text after the [tok] prefix, or None to abstain.
    call: {id, name, args}. result: None while running."""
    args = call["args"]
    args = list(args.values()) if isinstance(args, dict) else [args]
    spans = [(call["name"] + "(", "accent")]
    for i, a in enumerate(args):
        if i: spans.append((", ", "accent"))
        s = repr(a)
        spans.append((s[:40] + "..." if len(s) > 40 else s, "muted"))
    spans.append((")", "accent"))
    detail = tool_result_text(result)
    if detail and "\n" not in detail:
        spans.append((" " + detail, "muted"))
    return spans


_tool_token_cache = {}

def tool_call_tokens(ctx, call, result) -> int:
    """Estimated tokens of a tool call's args + result (incl. images). Cached per call id."""
    key = (call["id"], result is None)
    n = _tool_token_cache.get(key)
    if n is None:
        args = call["args"]
        s = args if isinstance(args, str) else json.dumps(args, default=str)
        n = get_token_estimate(s + tool_result_text(result), ctx.model)
        for a in getattr(result, "attachments", ()):
            if isinstance(a, ImageAttachment):
                n += min(1600, a.width * a.height // 750)
        _tool_token_cache[key] = n
    return n


def _fmt_tokens(n):
    return str(n) if n < 1000 else f"{round(n / 1000)}k"


def render_tool_line(buf, x, y, w, ctx, row, th):
    if row.status == 'running':
        label, color = f" {SPIN[int(time.time() * 8) % 4]} ", th.warning
    elif row.status == 'error':
        label, color = " x ", th.error
    else:
        n = row.tokens
        label = _fmt_tokens(n)
        color = th.muted if n < 500 else th.success if n < 1500 else th.warning if n <= 4000 else th.error
    buf.puts(x, y, f"[{label}]", txt_color=color, style='bold')
    col, end = x + 6, x + w
    for text, key in ctx.app.call("render_tool_call", ctx, row.call, row.result):
        s = text.replace('\r', ' ').replace('\n', ' ')[:max(0, end - col)]
        buf.put_cells(col, y, s, txt_color=getattr(th, key))
        col += len(s)




def _poll_operation(ctx, operation: Operation):
    """
    waits until operation is completed
    """
    value, canceled = None, False
    while value is None:
        if ctx.stop_early and not canceled:
            canceled = True
            try:
                operation.cancel()
            except Exception as e:
                ctx.app.debug_print(f"operation cancel failed: {e}")
        value = operation.poll(0.1)
    return value


@_overridable
def call_tools(ctx: Context, llm_result: LLMResult) -> bool:
    '''
    Blocks the thread until tools are complete.
    Returns whether the LLM should loop (True when there are tool-calls.)
    '''
    if not llm_result.tool_calls:
        return False

    tools = ctx.get_tools()
    pending = [PendingTool(tc) for tc in llm_result.tool_calls]
    threads = []

    def run_tool(record, fn):
        tc = record.call
        try:
            args = tc["args"]
            if 'tool_call_id' in inspect.signature(fn).parameters:
                args = {**args, 'tool_call_id': tc["id"]}
            value = fn(ctx, **_check_tool_args(fn, args))
            if isinstance(value, Operation):
                value = _poll_operation(ctx, value)
            if not isinstance(value, ToolResult):
                value = ToolResult(tool_result_text(value))
            if len(value.text) > MAX_TOOL_OUTPUT_CHARACTERS:
                value = ToolResult(f"ERROR: Tool output too large ({len(value.text)} chars, max {MAX_TOOL_OUTPUT_CHARACTERS}).")
        except Exception as e:
            ctx.app.debug_print(f"tool {tc['name']} failed: {e}")
            value = ToolResult(f"ERROR: {e}")
        record.result = value

    ctx.pending_tools = pending
    try:
        for record in pending:
            if ctx.stop_early: break
            tc = record.call
            fn = tools.get(tc["name"])
            if not fn:
                record.result = ToolResult(f"ERROR: Unknown tool: {tc['name']}")
                continue
            t = threading.Thread(target=run_tool, args=(record, fn))
            t.start()
            threads.append(t)
    finally:
        for t in threads:
            t.join()
        ctx.pending_tools = None

    with ctx._msg_lock:
        if ctx.stop_early:
            return False
        if ctx.pending_message is None:
            return True
        ctx._messages.extend([ctx.pending_message] + [
            Message(role="tool", content=p.result, tool_call_id=p.call["id"]) for p in pending
        ])
        ctx.pending_message = None
    return True



SIMPLE_DATA_TYPES = (str, int, float, bool, type(None))

class StrictDataDict(dict):
    """Dict that only allows str/int/float/bool/None values. Catches bugs early."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(dict(*args, **kwargs))
    def __setitem__(self, key, value):
        if not isinstance(value, SIMPLE_DATA_TYPES):
            raise TypeError(f"ctx.data[{key!r}] = {type(value).__name__} — only str/int/float/bool/None allowed. Use ctx.data_volatile for complex objects.")
        super().__setitem__(key, value)
    def update(self, other=(), **kwargs):
        if isinstance(other, dict): other = other.items()
        for k, v in other: self[k] = v
        for k, v in kwargs.items(): self[k] = v
    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default  # goes through __setitem__
        return self[key]


@dataclass
class Context:
    app: App
    name: str
    model: str
    reasoning: Literal["low","medium","high","none"] = "none"  # "low", "medium", "high", or "none"
    invoke_llm: Optional[Callable] = None  # explicit LLM backend; else the invoke_llm handler chain picks by model

    messages: InitVar[Optional[list[Message]]] = None
    tools: list[Callable] = field(default_factory=list)
    schema_id: Optional[str] = None
    max_tokens: int = 200000
    llm_is_running: bool = False
    pending_message: Message | None = field(default=None, init=False, repr=False)
    llm_cot: str = ""
    llm_output_type: str = ""
    last_invoke_time_end: float = 0
    last_invoke_time_start: float = 0
    last_seen: float = 0  # last frame the user had this context open in work mode
    llm_result: Optional[LLMResult] = None
    ui_stack: list = field(default_factory=list)
    llm_suspended: bool = False
    stop_early: bool = False
    parent: Optional[str] = None # name of parent context (for subagents)

    cwd: Optional[str] = None
    yolo: bool = False

    # data and volatile_data: Places where plugins can store arbitrary data.
    # Convention is to store data like ctx.data["pluginName:key"] = value
    data: StrictDataDict = field(default_factory=StrictDataDict) # str/int/float/bool/None only. Use data_volatile for complex objects.
    data_volatile: dict = field(default_factory=dict) # cleared on fork/clear. For complex/mutable objects.

    _context_hooks: dict = field(default_factory=lambda: {kind: [] for kind in CONTEXT_HOOK_KINDS}, init=False, repr=False)
    _messages: list = field(default_factory=list, init=False, repr=False)
    _msg_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _read_hashes: dict[str,str] = field(default_factory=dict) # file read tracking
    _line_snapshots: dict = field(default_factory=dict) # path -> {line_no: line_content}
    _prev_height: int = 0 # how many lines were used in rendering last frame
    pending_tools: list[PendingTool] | None = field(default=None, init=False, repr=False)
    _clear_pending: bool = field(default=False, init=False, repr=False)
    _scroll_up: int = 0
    _input_box: Optional['InputBox'] = None
    _scheduled: Optional[tuple] = field(default=None, init=False, repr=False)
    _pending_inputs: list = field(default_factory=list, init=False, repr=False)

    @property
    def context_hooks(self):
        return self._context_hooks

    def add_context_hook(self, kind, fn):
        self._context_hooks[kind].append(fn)
        return fn

    def output_renderer(self, fn):
        return self.add_context_hook("output_renderer", fn)

    def after_tool_calls(self, fn):
        return self.add_context_hook("after_tool_calls", fn)

    def after_llm_turn(self, fn):
        return self.add_context_hook("after_llm_turn", fn)

    def get_messages(self) -> tuple:
        with self._msg_lock:
            return tuple(self._messages)

    def edit_messages(self, fn):
        with self._msg_lock:
            fn(self._messages)

    def append_message(self, message: Message):
        with self._msg_lock:
            self._messages.append(message)

    def purge_cache(self):
        with self._msg_lock:
            for message in self._messages:
                message._snapshot = None

    def token_count(self) -> int:
        with self._msg_lock:
            total = 0
            for message in reversed(self._messages):
                if message._current_tokens is not None:
                    total += message._current_tokens
                    break
                content = tool_result_text(message.get_msg(self))
                total += get_token_estimate(content, self.model)

            if self.pending_message:
                total += get_token_estimate(self.pending_message.content + self.llm_cot, self.model)
            return total

    def is_token_count_estimate(self) -> bool:
        messages = self.get_messages()
        if not messages or messages[-1]._current_tokens is None:
            return True
        return self.pending_message is not None

    def __post_init__(self, messages):
        self.tools = list(self.tools)
        for fn in self.tools:
            _validate_tool_sig(fn.__name__, fn)
        for m in messages or ():
            # Copy defensively: messages are often shared module-level constants.
            m = copy.copy(m)
            m.tools = list(m.tools)
            m.attachments = dict(m.attachments)
            self._messages.append(m)
        if self.schema_id:
            self.app.context_schemas.setdefault(self.schema_id, self)

    def get_input_box(self):
        if self._input_box is None:
            self._input_box = InputBox(self.submit_input, allow_images=True,
                                       transform=lambda t: self.app.pipeline("transform_user_prompt", t, self))
        return self._input_box

    def submit_input(self, value: InputValue):
        if value.text.startswith("/"):
            if self.app.tui:
                self.app.dispatch_command(self.app.tui, value.text)
        elif not self.is_running():
            self.invoke(value.text, attachments=value.attachments)
        else:
            self._pending_inputs.append(value)
            self.stop_early = True

    def __hash__(self): return id(self)
    def __eq__(self, other): return self is other
    def is_running(self): return self.llm_is_running

    def mark_file_read(self, path, read_line_numbers=None):
        p = self.resolve(path)
        with open(p, "rb") as f:
            data = f.read()
        self._read_hashes[p] = hashlib.md5(data).hexdigest()
        if read_line_numbers is not None:
            lines = data.decode().splitlines()
            if p not in self._line_snapshots:
                self._line_snapshots[p] = {}
            self._line_snapshots[p].update({n: lines[n-1] for n in read_line_numbers})

    def get_line_snapshot(self, path):
        return self._line_snapshots.get(self.resolve(path), {})

    def resolve(self, path: str) -> str:
        """Resolve a path relative to this context's cwd."""
        if os.path.isabs(path):
            return os.path.normpath(path)
        base = self.cwd or os.getcwd()
        return os.path.normpath(os.path.join(base, path))

    def has_read_file(self, path):
        p = self.resolve(path)
        if p not in self._read_hashes:
            return False
        with open(p, "rb") as f:
            return self._read_hashes[p] == hashlib.md5(f.read()).hexdigest()

    def get_read_files(self):
        return list(self._read_hashes.keys())

    def with_tools(self, tools: list[Callable]) -> 'Context':
        for fn in tools:
            _validate_tool_sig(fn.__name__, fn)
        by_name = {fn.__name__: fn for fn in self.tools}
        by_name.update({fn.__name__: fn for fn in tools})
        clone = self.fork()
        clone.tools = list(by_name.values())
        return clone

    def without_tools(self, tools: list[Union[str, Callable]]) -> 'Context':
        names = {t if isinstance(t, str) else t.__name__ for t in tools}
        missing = names - {fn.__name__ for fn in self.tools}
        assert not missing, f"tools not present: {missing}"
        clone = self.fork()
        clone.tools = [fn for fn in self.tools if fn.__name__ not in names]
        return clone

    def get_tools(self) -> dict[str, Callable]:
        tools = {}
        for m in self.get_messages():
            for fn in m.tools:
                tools[fn.__name__] = fn
        # Context tools override message tools with the same name, preserving order.
        tools.update({fn.__name__: fn for fn in self.tools})
        return tools

    def get_tool_schemas(self) -> list[dict]:
        return [tool_to_schema(name, fn) for name, fn in self.get_tools().items() if not isinstance(fn, NativeTool)]

    def get_native_tools(self) -> list[str]:
        return [name for name, fn in self.get_tools().items() if isinstance(fn, NativeTool)]

    def schedule(self, fn: Callable, *args, **kwargs):
        """
        Wait for the model and its tools to finish, then call fn with the supplied arguments.
        fn is called immediately if the model is not running. 
        Only one function can be waiting;
        scheduling another raises an error. clear() cancels the waiting function.

        (Use this when you need the model to control it's own looping behaviour from within it's tool loop.)
        """
        with self._msg_lock:
            if self._clear_pending:
                raise RuntimeError("Clear is pending")
            if self._scheduled is not None:
                raise RuntimeError("A callback is already scheduled")
            if self.llm_is_running:
                self._scheduled = (fn, args, kwargs)
                return
        fn(*args, **kwargs)

    def _read_llm_stream(self, llm_fn):
        self.last_invoke_time_start = time.time()
        message = Message(role="assistant", content="")
        with self._msg_lock:
            self.pending_message = message
            self.llm_cot = self.llm_output_type = ""
            self.llm_result = None
        with closing(llm_fn(self)) as stream:
            for item in stream:
                if self.stop_early: return
                self._receive_stream_item(message, item)
        return message

    def _receive_stream_item(self, message: Message, item: ResponseChunk | LLMResult) -> None:
        with self._msg_lock:
            if isinstance(item, ResponseChunk):
                self.llm_output_type = item.type
                if item.type == "text":
                    message.content += item.content
                else: self.llm_cot += item.content
            elif isinstance(item, LLMResult):
                self.llm_result = item
                message.tool_calls = item.tool_calls
                message.native_calls = item.native_calls or None
                if item.input_tokens:
                    message._current_tokens = item.input_tokens + item.output_tokens

    def get_invoker(self) -> Callable:
        return self.invoke_llm or self.app.call("invoke_llm", self)

    def invoke(self, text=None, llm_fn=None, attachments=None):
        app = self.app
        llm_fn = llm_fn or self.get_invoker()
        app.debug_print(f"[invoke] {self.name} {self.model} -> {llm_fn.__module__}.{llm_fn.__name__}")
        with self._msg_lock:
            if self.llm_is_running:
                raise RuntimeError("Context is already running or draining")
            self.llm_is_running = True
            self.stop_early = False
            self.llm_result = None
            self.pending_message = None
            self.llm_cot = self.llm_output_type = ""
            self._scroll_up = 0
        def run():
            try:
                if text is not None:
                    self.append_message(Message(role="user", content=text, attachments=dict(attachments or {})))
                should_loop = True
                while should_loop and not self.stop_early and self._scheduled is None:
                    message = self._read_llm_stream(llm_fn)
                    if self.stop_early: break
                    if not message.tool_calls:
                        with self._msg_lock:
                            self.append_message(message)
                            self.pending_message = None
                    app.run_context_hooks("after_llm_turn", self)
                    if self.stop_early or not self.llm_result: break
                    self.llm_suspended = True
                    should_loop = app.call("call_tools", self, self.llm_result)
                    if should_loop and not self.stop_early:
                        app.run_context_hooks("after_tool_calls", self)
                    self.llm_suspended = False
            finally:
                with self._msg_lock:
                    self.llm_is_running = False
                    self.llm_suspended = False
                    self.pending_message = None
                    self.llm_cot = self.llm_output_type = ""
                    self.last_invoke_time_end = time.time()
                    scheduled = self._scheduled
                    self._scheduled = None
                    if self._clear_pending:
                        self.clear()
                        scheduled = None
                if scheduled is not None:
                    fn, args, kwargs = scheduled
                    fn(*args, **kwargs)
                elif self._pending_inputs:
                    next_input = self._pending_inputs.pop(0)
                    self.submit_input(next_input)

        threading.Thread(target=run, daemon=True).start()
    
    def truncate(self, index: int):
        """Remove all messages from index onward.
        Safe to call from tool execution (llm_suspended) or when LLM is not running.
        Raises if LLM is actively streaming."""
        with self._msg_lock:
            if self.llm_is_running and not self.llm_suspended:
                raise RuntimeError("Cannot truncate while LLM is streaming")
            del self._messages[index:]
            self.pending_message = None

    def clear(self):
        with self._msg_lock:
            self.stop_early = True
            self._scheduled = None
            if self.llm_is_running:
                self._clear_pending = True
                return
            self._clear_pending = False
            i = 0
            while i < len(self._messages) and self._messages[i].role == "system":
                i += 1
            del self._messages[i:]
            self.ui_stack = []
            self.llm_is_running = False
            self.llm_suspended = False
            self.llm_result = None
            self.pending_message = None
            self.llm_cot = self.llm_output_type = ""
            self.last_invoke_time_start = 0
            self.last_invoke_time_end = 0
            self._read_hashes = {}
            self._line_snapshots = {}
            self.data_volatile = {}

    def clone_with_context(self, serialized_context: str) -> 'Context':
        payload = json.loads(serialized_context)
        if payload.get("format") != 1:
            raise ValueError("Unsupported context format")
        if payload.get("schema_id") != self.schema_id:
            raise ValueError(f"Context schema mismatch: {payload.get('schema_id')!r}")
        tools = self.get_tools()
        messages = []
        for item in payload["messages"]:
            item = dict(item)
            item["tools"] = [tools[name] for name in item.get("tools", [])]
            messages.append(Message(**item))
        cpy = self._clone(payload.get("name") or self.name, messages)
        cpy.data = StrictDataDict(payload.get("data", {}))
        return cpy

    def _clone(self, new_name: str, messages: list[Message]) -> 'Context':
        cpy = copy.copy(self)
        cpy.tools = list(self.tools)
        cpy._msg_lock = threading.RLock()
        cpy._clear_pending = False
        cpy._pending_inputs = []
        cpy._context_hooks = {kind: list(fns) for kind, fns in self._context_hooks.items()}
        cpy._messages = messages
        cpy._read_hashes = dict(self._read_hashes)
        cpy._line_snapshots = {k: dict(v) for k, v in self._line_snapshots.items()}
        cpy.data = StrictDataDict(self.data)
        cpy.data_volatile = {}
        cpy.pending_tools = None
        cpy.pending_message = None
        cpy.llm_cot = cpy.llm_output_type = ""
        cpy.llm_result = None
        cpy.llm_suspended = False
        cpy._scheduled = None
        cpy.ui_stack = []
        cpy._input_box = None
        cpy.llm_is_running = False
        cpy.name = new_name
        cpy.parent = self.name
        return cpy

    def fork(self, new_name: Optional[str] = None) -> 'Context':
        messages = copy.deepcopy(self.get_messages())
        return self._clone(new_name or self.name, list(messages))

    def ask_human(self, draw):
        """Show a panel until draw(buf, inpt, r) returns a value (returned here). Blocks the calling (tool) thread.
        Returns None if the agent is stopped first."""
        result = []
        def panel(buf, inpt, r):
            value = draw(buf, inpt, r)
            if value is not None:
                result.append(value)
                self.ui_stack = [p for p in self.ui_stack if p is not panel]
        self.ui_stack.append(panel)
        while not result and not self.stop_early and panel in self.ui_stack:
            time.sleep(0.05)
        self.ui_stack = [p for p in self.ui_stack if p is not panel]
        return result[0] if result else None







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
        # double-buffer: prev holds last-flushed frame for diffing
        # _prev_chars uses sentinel to guarantee full first-frame draw (None != ' ')
        self._prev_chars: List[List[str]] = [['\x00'] * w for _ in range(h)]
        self._prev_styles: List[List[Optional[str]]] = [['\x00'] * w for _ in range(h)]
        self._prev_txt_colors: List[List[Optional[str]]] = [['\x00'] * w for _ in range(h)]
        self._prev_bg_colors: List[List[Optional[str]]] = [['\x00'] * w for _ in range(h)]
        self._invalidated = False

    def put(self, x, y, char, style=None, txt_color=None, bg_color=None):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.chars[y][x] = char
            self.styles[y][x] = style
            self.txt_colors[y][x] = txt_color
            self.bg_colors[y][x] = bg_color

    def puts(self, x, y, text, style=None, txt_color=None, bg_color=None):
        self.put_cells(x, y, text.replace('\r', ' ').replace('\n', ' '), style, txt_color, bg_color)

    def put_cells(self, x, y, cells, style=None, txt_color=None, bg_color=None):
        """Write a str (one char per cell) or list of cell strings at (x, y), clipped. No newline handling."""
        if not 0 <= y < self.h: return
        i, j = max(0, -x), min(len(cells), self.w - x)
        if j <= i: return
        a, b, n = x + i, x + j, j - i
        self.chars[y][a:b] = cells[i:j]
        self.styles[y][a:b] = [style] * n
        self.txt_colors[y][a:b] = [txt_color] * n
        self.bg_colors[y][a:b] = [bg_color] * n

    def invalidate(self):
        self._invalidated = True

    def clear_line(self, y):
        if not 0 <= y < self.h:
            return
        w = self.w
        self.chars[y][:] = [' '] * w
        self.styles[y][:] = [None] * w
        self.txt_colors[y][:] = [None] * w
        self.bg_colors[y][:] = [None] * w

    def clear(self):
        for y in range(self.h): self.clear_line(y)

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _style_wrap(term, fg, st, bg):
        """(prefix, suffix) escape sequences for a (fg, style, bg) run."""
        styled_fg = None
        if isinstance(fg, tuple):
            styled_fg = term.color_rgb(*fg)
            attr = st
        elif fg and st: attr = f"{fg}_{st}"
        elif fg: attr = fg
        elif st: attr = st
        else: attr = None
        if bg and isinstance(bg, tuple): styled_bg = term.on_color_rgb(*bg)
        elif bg: attr = f"{attr}_on_{bg}" if attr else f"on_{bg}"; styled_bg = None
        else: styled_bg = None
        styled = getattr(term, attr, None) if attr else None
        s = '\x01'
        if styled: s = styled(s)
        if styled_fg: s = styled_fg(s)
        if styled_bg: s = styled_bg(s)
        return tuple(s.split('\x01'))

    def flush(self, term, output=None):
        chars, styles, fgs, bgs = self.chars, self.styles, self.txt_colors, self.bg_colors
        pc, ps, pf, pb = self._prev_chars, self._prev_styles, self._prev_txt_colors, self._prev_bg_colors
        if self._invalidated:
            rows = range(self.h)
        else:  # whole-row compares run at C speed
            rows = [y for y in range(self.h) if chars[y] != pc[y] or fgs[y] != pf[y] or styles[y] != ps[y] or bgs[y] != pb[y]]
        self._invalidated = False

        out = []
        for y in rows:
            # Clear dirty rows first so content omitted by this frame cannot linger.
            out.append(term.move(y, 0))
            out.append(term.clear_eol)
            cr, sr, fr, br = chars[y], styles[y], fgs[y], bgs[y]
            end = self.w
            while end and cr[end - 1] == ' ' and sr[end - 1] is None and fr[end - 1] is None and br[end - 1] is None:
                end -= 1
            x = 0
            for key, grp in itertools.groupby(zip(fr[:end], sr[:end], br[:end])):
                n = sum(1 for _ in grp)
                pre, post = self._style_wrap(term, *key)
                out.append(pre + "".join(cr[x:x + n]).replace('\t', ' ') + post)
                x += n

        if out:
            output = output or sys.__stdout__
            output.write("".join(out))
            output.flush()
        # swap: current becomes prev, prev becomes current (will be cleared next frame)
        self.chars, self._prev_chars = pc, chars
        self.styles, self._prev_styles = ps, styles
        self.txt_colors, self._prev_txt_colors = pf, fgs
        self.bg_colors, self._prev_bg_colors = pb, bgs

    def fill(self, r: Rect, char='█', style=None, txt_color=None, bg_color=None):
        x, y, w, h = r
        for row in range(max(0, y), min(self.h, y + h)):
            self.put_cells(x, row, [char] * w, style, txt_color, bg_color)

    def rect(self, r: Rect, char='█', style=None, txt_color=None, bg_color=None):
        x, y, w, h = r
        self.put_cells(x, y, [char] * w, style, txt_color, bg_color)
        self.put_cells(x, y + h - 1, [char] * w, style, txt_color, bg_color)
        for row in range(y + 1, y + h - 1):
            self.put(x, row, char, style, txt_color, bg_color)
            self.put(x + w - 1, row, char, style, txt_color, bg_color)

    def rect_line(self, r: Rect, style=None, txt_color=None, bg_color=None):
        x, y, w, h = r
        if w < 2 or h < 2: return
        self.put_cells(x, y, '┌' + '─' * (w - 2) + '┐', style, txt_color, bg_color)
        self.put_cells(x, y + h - 1, '└' + '─' * (w - 2) + '┘', style, txt_color, bg_color)
        for row in range(y + 1, y + h - 1):
            self.put(x, row, '│', style, txt_color, bg_color)
            self.put(x + w - 1, row, '│', style, txt_color, bg_color)

    def hline(self, r: Rect, style=None, txt_color=None, bg_color=None):
        x, y, w, _ = r
        self.put_cells(x, y, '─' * w, style, txt_color, bg_color)

    def vline(self, r: Rect, style=None, txt_color=None, bg_color=None):
        x, y, _, h = r
        for i in range(h):
            self.put(x, y + i, '│', style, txt_color, bg_color)

    def print_wrapped(self, txt: str, x: int, y: int, w: int, style=None, txt_color=None, bg_color=None) -> int:
        """Write text with word-wrapping at width w. Returns number of lines consumed."""
        lines = txt.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        chars, styles, fgs, bgs, W, H = self.chars, self.styles, self.txt_colors, self.bg_colors, self.w, self.h
        row = 0
        for line in lines:
            if not line:
                row += 1; continue
            col = 0
            words = line.split(' ')
            last = len(words) - 1
            for wi, word in enumerate(words):
                L = len(word)
                if col > 0 and col + L > w:
                    row += 1; col = 0
                i = 0
                while i < L:  # long words hard-wrap
                    if col >= w: row += 1; col = 0
                    n = L - i if L - i < w - col else max(1, w - col)
                    yy, a = y + row, x + col
                    if 0 <= yy < H:  # clip [a, a+n) to [0, W)
                        s, e = (0 if a >= 0 else -a), (n if a + n <= W else W - a)
                        if s < e:
                            chars[yy][a + s:a + e] = word[i + s:i + e]
                            k = e - s
                            styles[yy][a + s:a + e] = [style] * k
                            fgs[yy][a + s:a + e] = [txt_color] * k
                            bgs[yy][a + s:a + e] = [bg_color] * k
                    col += n; i += n
                if wi < last and col < w:
                    col += 1  # space
            row += 1
        return max(row, 1)

    def fill_bg(self, x, y, w, h, color):
        a, b = max(0, x), min(self.w, x + w)
        if b <= a: return
        for yy in range(max(0, y), min(self.h, y + h)):
            row = self.bg_colors[yy]
            row[a:b] = [color if c is None else c for c in row[a:b]]

    def writer(self, x, y, w):
        return WrapWriter(self, x, y, w)

    def print_contained(self, txt: str, r: Rect, style=None, txt_color=None, bg_color=None, wrap=True, newlines=True) -> int:
        x, y, w, h = r
        if not newlines:
            txt = txt.replace('\n', ' ')
        txt = txt.replace('\r\n', '\n').replace('\r', '\n')
        row, col = 0, 0
        for li, line in enumerate(txt.split('\n')):
            if li: row += 1
            col = 0
            if not wrap:
                if row < h:
                    col = max(0, min(len(line), w))
                    self.put_cells(x, y + row, line[:col], style, txt_color, bg_color)
                continue
            i, L = 0, len(line)
            while i < L:
                if col >= w: row += 1; col = 0
                if row >= h: break
                n = max(1, min(L - i, w - col))
                self.put_cells(x + col, y + row, line[i:i + n], style, txt_color, bg_color)
                col += n; i += n
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

    def puts(self, cells, style=None, txt_color=None, bg_color=None):
        """Write a run of cells (str or list of cell strings), wrapping at w. Same result as put() per cell."""
        i, L = 0, len(cells)
        while i < L:
            if self.cx >= self.w:
                self.cy += 1; self.cx = 0
            n = max(1, min(L - i, self.w - self.cx))
            self.buf.put_cells(self.x + self.cx, self.y + self.cy, cells[i:i + n], style, txt_color, bg_color)
            self.cx += n; i += n

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
    KEY_ALIASES = {
        '\x17': ('KEY_CTRL_BACKSPACE',),
        '\x08': ('KEY_CTRL_BACKSPACE',),
        '\x1bd': ('KEY_CTRL_DELETE',),
        '\x18': ('KEY_CTRL_X',),
        '\x03': ('KEY_CTRL_C',),
        '\x1bv': ('KEY_ALT_V',),
    }

    def __init__(self, keys: list):
        self._keys = list(keys)

    def _candidates(self, k):
        s = str(k)
        aliases = self.KEY_ALIASES.get(s)
        if aliases:
            return {s, *aliases}
        return {k.name, s}

    def consume(self, *names: str) -> bool:
        wanted = set(names)
        for i, k in enumerate(self._keys):
            if self._candidates(k) & wanted:
                self._keys.pop(i)
                return True
        return False

    def peek(self, *names: str) -> bool:
        wanted = set(names)
        for k in self._keys:
            if self._candidates(k) & wanted:
                return True
        return False

    def consume_text(self) -> str:
        n_printable = sum(1 for k in self._keys if not k.is_sequence and str(k).isprintable())
        is_paste = n_printable > 1
        text = ""
        remaining = []
        for k in self._keys:
            s = str(k)
            if k.name == 'BRACKETED_PASTE':
                text += k.text.replace('\r\n', '\n').replace('\r', '\n')
            elif not k.is_sequence and s.isprintable():
                text += s
            elif is_paste and (s == '\n' or k.name == 'KEY_ENTER'):
                text += '\n'
            else:
                remaining.append(k)
        self._keys[:] = remaining
        return text



class InputBox:
    def __init__(self, on_submit=None, allow_images=False, transform=None):
        self.on_submit = on_submit
        self.allow_images = allow_images
        self.transform = transform  # fn(text) -> text on Enter; a change is shown first, next Enter sends
        self._shown = None  # last transform result, sent as-is
        self.text = ""
        self.cursor = 0
        self.attachments = {}

    def _prev_word(self, i):
        t = self.text
        while i > 0 and not t[i-1].isalnum(): i -= 1
        while i > 0 and t[i-1].isalnum(): i -= 1
        return i

    def _next_word(self, i):
        t = self.text
        while i < len(t) and t[i].isalnum(): i += 1
        while i < len(t) and not t[i].isalnum(): i += 1
        return i

    def _insert_attachment(self, attachment):
        token = f"[pasted-image 0x{Path(attachment.path).stem[:5]}]"
        n = 2
        while token in self.text or token in self.attachments:
            token = f"[pasted-image 0x{Path(attachment.path).stem[:5]}-{n}]"
            n += 1
        self.attachments[token] = attachment
        self.text = self.text[:self.cursor] + token + self.text[self.cursor:]
        self.cursor += len(token)

    @staticmethod
    def _wrap(s, w):
        lines = []
        for part in s.split('\n'):
            if not part:
                lines.append('')
            else:
                lines.extend(part[i:i + w] for i in range(0, len(part), w))
        return lines

    def _cursor_visual(self, w):
        lines = self._wrap(self.text[:self.cursor], w)
        if not lines: return 0, 0
        return len(lines) - 1, len(lines[-1])

    def __call__(self, buf: ScreenBuffer, inpt, r, txt_color='white'):
        return self.draw(buf, inpt, r, txt_color=txt_color)

    def draw(self, buf: ScreenBuffer, inpt, r, txt_color='white'):
        """Returns the InputValue submitted this frame (also passed to on_submit), else None."""
        inner_w = r[2]
        if inner_w < 1: return None

        if self.allow_images and inpt.consume('KEY_ALT_V'):
            attachment = clipboard_image_attachment()
            if attachment:
                self._insert_attachment(attachment)

        typed = inpt.consume_text()
        if typed:
            self.text = self.text[:self.cursor] + typed + self.text[self.cursor:]
            self.cursor += len(typed)
        for i, k in enumerate(inpt._keys):
            if k.name == 'KEY_ALT_ENTER' or str(k) == '\n':
                self.text = self.text[:self.cursor] + '\n' + self.text[self.cursor:]
                self.cursor += 1
                inpt._keys.pop(i)
                break
        if inpt.consume('KEY_LEFT') and self.cursor > 0: self.cursor -= 1
        if inpt.consume('KEY_RIGHT') and self.cursor < len(self.text): self.cursor += 1
        if inpt.consume('KEY_BACKSPACE') and self.cursor > 0:
            self.text = self.text[:self.cursor-1] + self.text[self.cursor:]
            self.cursor -= 1
        if inpt.consume('KEY_DELETE') and self.cursor < len(self.text):
            self.text = self.text[:self.cursor] + self.text[self.cursor+1:]
        if inpt.consume('KEY_CTRL_BACKSPACE') and self.cursor > 0:
            nc = self._prev_word(self.cursor)
            self.text = self.text[:nc] + self.text[self.cursor:]
            self.cursor = nc
        if inpt.consume('KEY_CTRL_DELETE') and self.cursor < len(self.text):
            self.text = self.text[:self.cursor] + self.text[self._next_word(self.cursor):]
        if inpt.consume('KEY_CTRL_LEFT'): self.cursor = self._prev_word(self.cursor)
        if inpt.consume('KEY_CTRL_RIGHT'): self.cursor = self._next_word(self.cursor)
        if inpt.consume('KEY_HOME'):
            cy, cx = self._cursor_visual(inner_w)
            self.cursor -= cx
        if inpt.consume('KEY_END'):
            cy, cx = self._cursor_visual(inner_w)
            line = self._wrap(self.text, inner_w)[cy]
            self.cursor += len(line) - cx
        if self.text and inpt.consume('KEY_ENTER'):
            if self.transform and self.text != self._shown:
                new = self.transform(self.text)
                if new != self.text:
                    self.text = self._shown = new
                    self.cursor = len(new)
                    return None
            attachments = {token: attachment for token, attachment in self.attachments.items()
                           if token in self.text}
            value = InputValue(self.text, attachments)
            self.text, self.cursor, self.attachments, self._shown = "", 0, {}, None
            if self.on_submit: self.on_submit(value)
            return value

        cursor_char = "█"
        cy, cx = self._cursor_visual(inner_w)
        visual_lines = self._wrap(self.text, inner_w)
        if not visual_lines: visual_lines = ['']

        max_visible = r[3]
        scroll = max(0, cy - max_visible + 1)
        for i, line in enumerate(visual_lines[scroll:scroll + max_visible]):
            vi = i + scroll
            if vi == cy:
                buf.puts(r[0], r[1]+i, line[:cx] + cursor_char + line[cx:], txt_color=txt_color)
            else:
                buf.puts(r[0], r[1]+i, line, txt_color=txt_color)

    def get_height(self, width):
        if width < 1: return 1
        return len(self._wrap(self.text, width)) if self.text else 1

    def set_text(self, t):
        self.text = t
        self.cursor = len(t)
        self.attachments = {}

    def get_text(self):
        return self.text.strip()


@_overridable
def make_input(on_submit=None):
    return InputBox(on_submit)



# --- SELECTION MODE UI ---


@_overridable
def render_selection_mode_context_name(tui, buf, ctx, x, y):
    th = tui.app.theme
    selected = ctx is tui.current
    approx = "~" if ctx.is_token_count_estimate() else ""
    toks = f" ({approx}{ctx.token_count()//1000}k)"
    spin = "/—\\|"[int(time.time() * 12) % 4]
    suffix = f" {spin}" if ctx.is_running() else ""
    if ctx.ui_stack: suffix += " ?"  # needs a human
    elif tui.is_unseen(ctx): suffix += " ●"  # finished while you weren't looking
    if ctx.yolo: suffix += " !"

    if ctx.is_running(): name_color = th.running
    elif ctx.llm_suspended: name_color = th.success
    else: name_color = th.text

    buf.puts(x, y, ctx.name, txt_color=name_color, style='bold' if selected else None)
    x += len(ctx.name)
    buf.puts(x, y, toks, txt_color=th.muted)
    x += len(toks)
    buf.puts(x, y, suffix, txt_color=th.warning)

@_overridable
def render_selection_left(tui, buf, inpt, r):
    x, y, w, h = r
    th = tui.app.theme
    buf.rect_line(r, txt_color=th.accent)

    ctxs = tui.context_list()
    idx = next((i for i, c in enumerate(ctxs) if c is tui.current), 0)

    for i, ctx in enumerate(ctxs):
        if i >= h - 2: break
        selected = (ctx is tui.current)
        row = y + 1 + i
        num = f"{i + 1:>3} " if selected else f"{abs(i - idx):>3} "  # relativenumber: `3j` at a glance
        buf.puts(x + 1, row, num, txt_color=th.selection if selected else th.muted, style='bold' if selected else None)
        tui.app.call("render_selection_mode_context_name", tui, buf, ctx, x + 1 + len(num), row)

@_overridable
def render_context_window_bar(tui, buf, ctx, x, y, bar_w):
    th = tui.app.theme
    token_count = ctx.token_count()
    ratio = min(token_count / ctx.max_tokens, 1) if ctx.max_tokens else 0
    color = th.accent_alt if ratio < 0.5 else th.warning if ratio < 0.8 else th.error
    cells = ratio * bar_w
    filled = int(cells)
    bar = "█" * filled
    if filled < bar_w and cells - filled >= 0.5: bar += "▌"
    buf.puts(x, y, bar, txt_color=color)
    buf.puts(x + len(bar), y, "░" * (bar_w - len(bar)), txt_color=th.muted)
    approx = "~" if ctx.is_token_count_estimate() else ""
    used = f" {approx}{token_count//1000}k"
    buf.puts(x + bar_w, y, used, txt_color=color)
    buf.puts(x + bar_w + len(used), y, f" / {ctx.max_tokens//1000}k  {ratio:.0%}", txt_color=th.muted)

@_overridable
def render_selection_right(tui, buf, r):
    x, y, w, h = r
    th = tui.app.theme
    buf.rect_line(r, txt_color=th.accent)

    ctx = tui.current
    buf.puts(x + 2, y + 1, ctx.name, txt_color=th.accent_alt, style='bold')
    model_x = x + 2 + len(ctx.name) + 2
    try: via = ctx.get_invoker().__module__.rsplit(".", 1)[-1]
    except RuntimeError: via = "no provider"
    model = ctx.model or ""
    buf.puts(model_x, y + 1, model, txt_color=th.text)
    buf.puts(model_x + len(model), y + 1, f"  · {via}", txt_color=th.muted)
    budget = tui.app.budget
    cost_str = f"${budget.get_cost():.2f}"
    limit_str = f" / ${budget.get_limit():.2f}"
    budget_x = x + w - len(cost_str) - len(limit_str) - 2
    buf.puts(budget_x, y + 1, cost_str, txt_color=th.error if budget.is_over() else th.success)
    buf.puts(budget_x + len(cost_str), y + 1, limit_str, txt_color=th.muted)

    bar_w = min(w - 4, 20)
    tui.app.call("render_context_window_bar", tui, buf, ctx, x + 2, y + 2, bar_w)

    buf.hline((x + 1, y + 3, w - 2, 1), txt_color=th.accent)

    msgs_end_x = x + w - 2
    tools = [f"{name} (native)" if isinstance(fn, NativeTool) else name for name, fn in ctx.get_tools().items()]
    if tools:
        msgs_end_x = _render_tools_column(buf, r, tools, th)

    groups = _group_agent_turns(ctx.get_messages())
    if not groups:
        buf.puts(x + 2, y + 4, "(no messages)", style='dim')
        return

    rows = []
    for group in groups:
        rows.extend(_overview_rows(group, th))

    last_row_y = y + h - 2
    for row_y, (segments, bg) in enumerate(rows, start=y + 4):
        if row_y > last_row_y: break
        buf.puts(x + 1, row_y, " " * (msgs_end_x - x - 1), bg_color=bg)
        seg_x = x + 2
        for text, color in segments:
            text = text[:max(0, msgs_end_x - seg_x)]
            buf.puts(seg_x, row_y, text, txt_color=color, bg_color=bg)
            seg_x += len(text)


def _render_tools_column(buf, r, tools, th):
    """Draw tool names on the right edge, behind a vertical separator. Returns the x where messages must stop."""
    x, y, w, h = r
    names_w = min(max(len(t) for t in tools), (w - 4) // 3)
    names_x = x + w - 2 - names_w
    sep_x = names_x - 2
    buf.vline((sep_x, y + 4, 1, h - 5), txt_color=th.accent)
    buf.put(sep_x, y + 3, '┬', txt_color=th.accent)
    buf.put(sep_x, y + h - 1, '┴', txt_color=th.accent)
    for i, name in enumerate(tools[:h - 5]):
        buf.puts(names_x, y + 4 + i, name[:names_w], txt_color=th.accent_alt, style='bold')
    return sep_x - 1


def _group_agent_turns(msgs):
    """Collapse runs of consecutive assistant/tool messages into lists; other messages stay as-is."""
    groups = []
    for msg in msgs:
        is_agent_msg = msg.role in ("assistant", "tool")
        if not is_agent_msg:
            groups.append(msg)
        elif groups and isinstance(groups[-1], list):
            groups[-1].append(msg)
        else:
            groups.append([msg])
    return groups


def _overview_rows(group, th):
    """Rows of (segments, bg) for one group; each segment is (text, color)."""
    if isinstance(group, list):
        return _agent_turn_rows(group, th)
    if group.role == "user":
        return [([("  " + _one_line(_msg_text(group)), th.muted)], th.user_background)]
    label = group.overview or group.role
    return [([(label, th.accent), (" " + _tokstr([group]), th.muted)], None)]


def _agent_turn_rows(msgs, th):
    turns = sum(m.role == "assistant" for m in msgs)
    label = f"assistant ×{turns}" if turns > 1 else "assistant"

    call_counts = Counter(tc["name"] for m in msgs for tc in m.tool_calls or [])
    calls_str = " ".join(f"{name}×{n}" if n > 1 else name for name, n in call_counts.items())

    summary = [(label, th.success), (" " + _tokstr(msgs), th.muted), ("  " + calls_str, th.muted)]
    rows = [(summary, None)]

    last = msgs[-1]
    if last.role != "assistant":
        return rows
    final_text = _one_line(_msg_text(last))
    if final_text:
        rows.append(([(">" + final_text, th.text)], None))
    return rows


def _one_line(text):
    return " ".join((text or "").split())


def _msg_text(msg):
    """Message text without evaluating callables; None if not yet snapshotted."""
    if callable(msg.content):
        return msg._snapshot
    return tool_result_text(msg.content)


def _tokstr(msgs):
    texts = [_msg_text(m) for m in msgs]
    if all(t is None for t in texts):
        return "(func)"
    toks = sum(get_token_estimate(t) for t in texts if t is not None)
    return f"(~{toks//1000}K)" if toks > 999 else f"(~{toks})"



def _tool_row(ctx, tc, result):
    if result is None:
        status = 'running'
    else:
        status = 'error' if tool_result_text(result).startswith("ERROR:") else 'ok'
    return ToolCall(call=tc, result=result, status=status, tokens=tool_call_tokens(ctx, tc, result))



@_overridable
def render_work_mode(tui, buf, inpt, r):
    x, y, w, h = r
    ctx = tui.current
    th = tui.app.theme

    with ctx._msg_lock:
        messages = list(ctx._messages)
        pending = ctx.pending_tools
        pending_assistant = ctx.pending_message
        if pending_assistant is not None:
            messages.append(pending_assistant)
    tool_results = {m.tool_call_id: m.content for m in messages if m.role == "tool"}

    message_outputs = []
    for msg in messages:
        if msg.role == "tool":
            continue
        text = tool_result_text(msg.get_msg(ctx))
        if msg is pending_assistant and ctx.is_running() and not ctx.llm_suspended and ctx.llm_result is None:
            cot = ctx.llm_cot if tui.show_cot else ""
            if cot:
                message_outputs.append(('cot', (cot + ("" if text else "█")).split('\n'), False))
            if cot and not text: continue
            text += "█"
        cache = msg._render_cache
        if cache and cache[0] == text and cache[1] is th:
            lines = list(cache[2])
        else:
            lines = text.split('\n')
            tui.app.run_context_hooks("output_renderer", ctx, lines, msg)
            msg._render_cache = (text, th, list(lines))
        lines[:0] = [_tool_row(ctx, {**nc, "name": "native:" + nc["name"]}, nc["result"]) for nc in msg.native_calls or ()]
        if msg is pending_assistant and pending is not None:
            lines.extend(_tool_row(ctx, p.call, p.result) for p in pending)
        elif msg.tool_calls:
            lines.extend(_tool_row(ctx, tc, tool_results.get(tc["id"])) for tc in msg.tool_calls)
        message_outputs.append((msg.role, lines, bool(msg.tool_calls)))


    available = h - 1
    ctx._scroll_up = min(ctx._scroll_up, max(0, ctx._prev_height - available))

    scroll_offset = max(0, ctx._prev_height - available) - ctx._scroll_up
    row = (y + 1) - scroll_offset
    prev_role = None
    for role, lines, has_tool_calls in message_outputs:
        if role == 'user' or prev_role == 'user': row += 1
        prev_role = role
        bg = th.user_background if role == 'user' else None
        fg = th.cot if role == 'cot' else th.text
        for line in lines:
            if line == '' and has_tool_calls: continue
            if isinstance(line, ToolCall):
                render_tool_line(buf, x, row, w, ctx, line, th)
                drawn = 1
            elif callable(line):
                drawn = line(buf, x, row, w)
            else:
                drawn = buf.print_wrapped(line, x, row, w, txt_color=fg, bg_color=bg)
            if bg: buf.fill_bg(x, row, w, drawn, bg)
            row += drawn
    if ctx.llm_result and ctx.llm_result.error:
        row += 1
        for line in ctx.llm_result.error.split('\n'):
            drawn = buf.print_wrapped(line, x, row, w, txt_color=th.text, bg_color=th.error_bg)
            buf.fill_bg(x, row, w, drawn, th.error_bg)
            row += drawn
    ctx._prev_height = row - (y - scroll_offset)

    if ctx._scroll_up > 0 and ctx._prev_height > available:
        total = ctx._prev_height
        bar_h = max(1, available * available // total)
        bar_top = (y + 1) + (available - bar_h) * (total - available - ctx._scroll_up) // (total - available)
        for sy in range(bar_top, bar_top + bar_h):
            buf.put(x + w - 1, sy, '█', txt_color=th.muted)

    buf.clear_line(y)
    label = f"{ctx.name}  {ctx.model or ''}"
    label_w = len(label) + 2
    bar_w = min(w - label_w - 20, 30)
    buf.puts(x, y, label, txt_color=th.text)
    tui.app.call("render_context_window_bar", tui, buf, ctx, x + label_w, y, bar_w)


@_overridable
def render_work_mode_input(tui, buf, inpt, input_r, input_box):
    ctx = tui.current
    th = tui.app.theme
    for y in range(input_r[1], input_r[1] + input_r[3]):
        buf.clear_line(y)
    if ctx.is_running():
        input_box(buf, inpt, input_r, txt_color=th.accent)
        spin = "[" + "/—\\|"[int(time.time() * 5) % 4] + "]"
        elapsed = f"{time.time() - ctx.last_invoke_time_start:.1f}s"
        if ctx.pending_message and ctx.llm_output_type:
            toks = f"~{get_token_estimate(ctx.pending_message.content + ctx.llm_cot, ctx.model)}"
            if ctx.llm_result:
                toks = str(ctx.llm_result.output_tokens)
            label = "thinking..." if ctx.llm_output_type == "cot" else "outputting..."
            stats = f" ({toks} toks, {elapsed}) "
        else:
            label = "invoking..."
            stats = f" ({elapsed}) "
        total_w = 4 + len(label) + len(stats)
        x = input_r[0] + input_r[2] - total_w
        y = input_r[1] - 1 if input_box.text else input_r[1]  # onto divider while typing
        buf.puts(x, y, spin, txt_color=th.invoking); x += 4
        buf.puts(x, y, label, txt_color=th.accent); x += len(label)
        buf.puts(x, y, stats, txt_color=th.muted)
        if ctx._pending_inputs:
            qtxt = ctx._pending_inputs[0].text.replace('\r', ' ').replace('\n', ' ')
            max_w = input_r[2] - total_w - 2
            prefix = "queued: \""
            if len(qtxt) > max_w - len(prefix) - 1:
                qtxt = qtxt[:max(0, max_w - len(prefix) - 4)] + "..."
            buf.puts(input_r[0] + 1, y, prefix + qtxt + "\"", txt_color=th.accent_alt)
    else:
        input_box(buf, inpt, input_r)




@_handleable
def transform_user_prompt(text, ctx):
    """Pipeline, run on input box Enter. A changed prompt is shown to the user first; the next Enter sends it."""
    return text


def request_approval(ctx, tool, args):
    """Run the approve_tool chain. True = approved, "reason" = denied, None = no handler decided."""
    if ctx.stop_early:
        return "stopped"
    return ctx.app.call("approve_tool", ctx, tool, args)

@_handleable
def approve_tool(ctx, tool, args):
    """Handlers return True (allow), "reason" (deny), or None (abstain)."""
    return None

def _approve_yolo(ctx, tool, args):
    if ctx.yolo: return True

@_handleable
def render_work_mode_footer(tui, buf, r, ctx):
    x, y, w, h = r
    th = tui.app.theme
    on = ctx.yolo
    buf.puts(x, y, "  yolo ON" if on else "  yolo OFF",
                txt_color=th.success if on else th.muted)

def render_workmodefooter_and_commands(tui, buf, r, ctx):
    th = tui.app.theme
    x, y, w, h = r
    for row in range(y, y + h):
        buf.clear_line(row)
    text = ctx.get_input_box().get_text()
    """
    todo: in future, instead of just "yolo ON" and "yolo OFF",
    we should have actual 'modes' here.
    Like, displaying auxiliary information about maybe like, task confidence,
    or some other stuff, maybe like `plan-mode ON`, or even some other shit.

    Do some thinking; we can do a lot more than this besides just yolo ON/OFF.
    """

    if not text.startswith("/"):
        tui.app.call("render_work_mode_footer", tui, buf, r, ctx)
        return

    query = text[1:].split(" ")[0].lower()
    names = sorted(tui.app.commands)
    prefix = [n for n in names if n.startswith(query)]
    fuzzy = difflib.get_close_matches(query, [n for n in names if n not in prefix], n=3)
    matches = (prefix + fuzzy)[:3]
    if not matches:
        buf.puts(x, y, "  no matching command", txt_color=th.muted)
        return

    for i, name in enumerate(matches):
        if i >= h:
            break
        fn, spec = tui.app.commands[name]
        args = " ".join(f"<{a}>" for a, _, _ in spec)
        doc = " ".join((fn.__doc__ or "").split())
        cx = x
        for s, col in (("  /", th.muted), (name, th.accent_alt),
                       (" " + args if args else "", th.warning),
                       ("  " + doc if doc else "", th.muted)):
            buf.puts(cx, y + i, s, txt_color=col); cx += len(s)


@_overridable
def render_status_line(tui, buf, inpt, r):
    """Bottom row: [MODE] agent info, echo, pending keys + which-key popup."""
    x, y, w, h = r
    th = tui.app.theme
    buf.clear_line(y)
    ctx = tui.current
    label, bg = (" AGENTS ", th.accent) if tui.mode == "selection" else (" WORK ", th.success)
    buf.puts(x, y, label, txt_color="black", bg_color=bg, style='bold')
    cx = x + len(label) + 1
    if ctx:
        info = f"{ctx.name} · {ctx.reasoning}" + (" · yolo" if ctx.yolo else "")
        buf.puts(cx, y, info, txt_color=th.muted)
        cx += len(info) + 2
    msg, t, err = tui.status
    if msg and time.time() - t < 4:
        buf.puts(cx, y, _one_line(msg)[:max(0, w - cx - 12)], txt_color=th.error if err else th.text)
    pending = "".join(tui.pending_keys)
    if not pending: return
    buf.puts(x + w - len(pending) - 1, y, pending, txt_color=th.accent_alt, style='bold')
    rows = tui.key_continuations()[:max(1, buf.h - 4)]
    if not rows: return
    kw = max(len(k) for k, _ in rows)
    pw = min(w - 2, max(kw + 3 + len(d) for k, d in rows) + 4)
    px, py = x + w - pw - 1, y - len(rows) - 2
    buf.fill((px, py, pw, len(rows) + 2), ' ')
    buf.rect_line((px, py, pw, len(rows) + 2), txt_color=th.accent)
    for i, (k, d) in enumerate(rows):
        buf.puts(px + 2, py + 1 + i, k.ljust(kw), txt_color=th.accent_alt, style='bold')
        buf.puts(px + 3 + kw, py + 1 + i, _one_line(d)[:pw - kw - 5], txt_color=th.text)


_ex6_dir = os.path.dirname(os.path.abspath(__file__))


_BOOTSTRAP_PLUGIN = '''\
import ex6
from _ex6.agents import auto_setup


def setup(app: ex6.App):
    auto_setup(app)
'''

def bootstrap():
    """Create _ex6/my_project.py and pyrightconfig.json in cwd (existing files left alone); gitignore the context dump."""
    Path("_ex6").mkdir(exist_ok=True)
    plugin, pyright, gitignore = Path("_ex6/my_project.py"), Path("pyrightconfig.json"), Path(".gitignore")
    if not plugin.exists():
        plugin.write_text(_BOOTSTRAP_PLUGIN)

    if not pyright.exists():
        pyright.write_text(json.dumps({"extraPaths": [_ex6_dir, os.path.join(_ex6_dir, "_ex6")]}, indent=4))

    if gitignore.exists() and ".EX6_CONTEXT_DUMP.txt" not in gitignore.read_text(encoding="utf-8").splitlines():
        with gitignore.open("a", newline="") as f:
            f.write("\n.EX6_CONTEXT_DUMP.txt\n")



def _load_plugins(app):
    import importlib.util
    import types

    # Core plugins: next to ex6.py. User plugins: in get_folder(). Project plugins: in cwd.
    core_dir = os.path.join(_ex6_dir, "_ex6")
    user_dir = str(get_folder() / "_ex6")
    project_dir = os.path.abspath("_ex6")
    dirs = []
    for d in (core_dir, user_dir, project_dir):
        if os.path.isdir(d) and d not in dirs: dirs.append(d)

    if not dirs: return

    def _register_pkg(name, dir_path):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [dir_path]
            sys.modules[name] = pkg
        else:
            # Add path so both core and project subpackages resolve
            existing = sys.modules[name]
            if dir_path not in existing.__path__:
                existing.__path__.append(dir_path)

    modules = []
    for plugin_dir in dirs:
        _register_pkg("_ex6", plugin_dir)
        for dirpath, dirnames, filenames in os.walk(plugin_dir):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(("_", ".")))
            rel = os.path.relpath(dirpath, plugin_dir).replace(os.sep, ".")
            pkg_name = "_ex6" if rel == "." else f"_ex6.{rel}"
            _register_pkg(pkg_name, dirpath)
            for filename in sorted(filenames):
                if not filename.endswith(".py") or filename.startswith("_"):
                    continue
                mod_name = f"{pkg_name}.{filename[:-3]}"
                path = os.path.join(dirpath, filename)
                existing = sys.modules.get(mod_name)
                existing_file = getattr(existing, '__file__', None)
                if existing_file:
                    if os.path.normcase(os.path.abspath(existing_file)) != os.path.normcase(os.path.abspath(path)):
                        app.debug_print(f"[plugins] skipped {path}: {mod_name} already loaded from {existing_file}")
                    module = existing
                else:
                    spec = importlib.util.spec_from_file_location(mod_name, path)
                    assert spec and spec.loader
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = module
                    spec.loader.exec_module(module)
                if module not in modules:
                    modules.append(module)

    for module in modules:
        app._register_module(module)
    for module in modules:
        app.setup_module(module)



@dataclass
class TUI:
    app: App
    contexts: list[Context] = field(default_factory=list, init=False)
    current: Optional[Context] = field(default=None, init=False)
    mode: Literal["selection", "work", "scroll"] = "selection"
    prev_mode: Literal["selection", "work", "scroll"] = "selection"
    term: Any = None
    ui_panel_stack: list = field(default_factory=list)
    sel_input: bool = False  # selection mode: current's input box is open (space)
    pending_keys: list = field(default_factory=list)  # tokens typed so far for a multi-key binding
    status: tuple = ("", 0.0, False)  # (msg, time, is_error), see echo()
    favorite: Optional[Context] = None  # template for new agents (`n`)
    buf: Any = None
    keyls: list = field(default_factory=list)
    input_box: Any = None  # per-context; resolved each frame
    stdout_sink: Any = None
    show_cot: bool = True
    log_keys: bool = False

    def add_context(self, ctx):
        if ctx.app is not self.app:
            raise RuntimeError("Context belongs to another app")
        if ctx not in self.contexts:
            self.contexts.append(ctx)
        if self.current is None:
            self.current = ctx
        return ctx

    def get_context(self, name):
        return next((c for c in self.contexts if c.name == name), None)

    def context_list(self):
        """Contexts in display order (selection list, `j/k`)."""
        return sorted(self.contexts, key=lambda c: c.name)

    def is_unseen(self, ctx):
        return not ctx.is_running() and ctx.last_invoke_time_end > ctx.last_seen

    def echo(self, msg, error=False):
        """Show a message in the status line for a few seconds."""
        self.status = (msg, time.time(), error)

    def keys_active(self):
        return self.mode == "selection" and self.current and not self.sel_input and not self.ui_panel_stack

    def _resolve_keys(self):
        """(fn, count) on a full match, "wait" on a prefix or count, None on no match."""
        toks = self.pending_keys
        i = 0
        while i < len(toks) and toks[i].isdigit() and (i or toks[i] != "0"): i += 1
        digits, seq = "".join(toks[:i]), tuple(toks[i:])
        if not seq: return "wait"
        if seq in self.app.keys:
            return self.app.keys[seq], int(digits or 1)
        return "wait" if any(b[:len(seq)] == seq for b in self.app.keys) else None

    def key_continuations(self):
        """[(keys, description)] for bindings continuing pending_keys (which-key popup)."""
        seq = tuple(t for t in self.pending_keys if not t.isdigit())
        if not seq: return []
        return [("".join(b[len(seq):]), (fn.__doc__ or "").strip().split("\n")[0])
                for b, fn in self.app.keys.items() if len(b) > len(seq) and b[:len(seq)] == seq]

    def feed_keys(self, inpt):
        """Run key bindings. Unbound single keys stay in inpt for others (eg arrow nav)."""
        keys, inpt._keys = inpt._keys, []
        for i, k in enumerate(keys):
            if not self.keys_active():  # a binding opened an input: the rest is typing
                inpt._keys += keys[i:]
                return
            self.pending_keys.append(key_token(k))
            r = self._resolve_keys()
            if r == "wait": continue
            if r is None and len(self.pending_keys) == 1: inpt._keys.append(k)
            self.pending_keys = []
            if r: r[0](self, r[1])

    def remove_context(self, ctx):
        if ctx not in self.contexts:
            return
        self.contexts.remove(ctx)
        if self.current is ctx:
            self.current = self.contexts[0] if self.contexts else None
        self.app.run_context_hooks("context_removed", ctx)

    def __post_init__(self):
        from blessed import Terminal
        self.app.tui = self
        self.term = Terminal()
        if os.name == "nt":
            import msvcrt
            def getch(decode_latin1=False):
                # HACK/MONKEYPATCH: handle invalid chars in terminal.
                char = msvcrt.getwch()
                if char in {'\x00', '\xe0'}:
                    char += msvcrt.getwch()
                return char
            self.term.getch = getch
        self.buf = ScreenBuffer(self.term.width, self.term.height)
        self.stdout_sink = _StdoutSink(self.app.debug_print)

    def enter_scroll_mode(self):
        if self.mode == "scroll":
            return
        self.prev_mode = self.mode
        self.mode = "scroll"
        sys.stdout, sys.stderr = self.app.stdout, self.app.stderr
        self.app.stdout.write(self.term.exit_fullscreen)
        self.app.stdout.flush()


def _tui_loop(tui: TUI):
    buf = tui.buf
    _sink = tui.stdout_sink
    term = tui.term

    if tui.mode != "scroll":
        sys.stdout, sys.stderr = _sink, _sink
    if tui.app.fatal_error:
        raise RuntimeError(f"FATAL: {tui.app.fatal_error}")
    for i in range(4):
        if i: time.sleep(0.001)  # inkey(timeout>0) sleeps in 10ms steps on Windows
        try:
            key = term.inkey(timeout=0, esc_delay=ESC_DELAY)
        except UnicodeDecodeError:
            key = None
        while key:
            if tui.log_keys:
                tui.app.debug_print(f"key: name={key.name!r} str={str(key)!r} code={key.code!r} seq={key.is_sequence}")
            tui.keyls.append(key)
            try:
                key = term.inkey(timeout=0, esc_delay=ESC_DELAY)
            except UnicodeDecodeError:
                key = None
        if tui.keyls:
            break

    if buf.w != term.width or buf.h != term.height:
        buf = tui.buf = ScreenBuffer(term.width, term.height)

    inpt = InputPass(tui.keyls)
    tui.keyls = []

    buf.clear()

    if not tui.contexts:
        th = tui.app.theme
        msg = "You must create a plugin with Contexts for ex6 to work."
        mx = (term.width - len(msg)) // 2
        my = term.height // 2
        buf.puts(mx, my, msg, txt_color=th.error)
        buf.flush(term, tui.app.stdout)
        return

    if tui.current not in tui.contexts:
        tui.current = tui.contexts[0]
    input_box = tui.input_box = tui.current.get_input_box()

    prev_mode = tui.mode

    # Clipboard cleaning: collapse consecutive non-indentation spaces every frame.
    # See CLIPBOARD_CLEANING and _try_clean_clipboard() for details.
    if CLIPBOARD_CLEANING:
        _try_clean_clipboard()

    if tui.ui_panel_stack and inpt.consume('KEY_ESCAPE'):
        tui.ui_panel_stack.pop()
    if tui.sel_input and inpt.consume('KEY_ESCAPE'):
        tui.sel_input = False
    if tui.keys_active():
        tui.feed_keys(inpt)
    input_box = tui.input_box = tui.current.get_input_box()  # a key may have switched agent

    footer_r = Region()
    status_r = Region(0, term.height - 1, term.width, 1)

    if tui.mode == "work":
        tui.current.last_seen = time.time()
        input_h = max(1, min(input_box.get_height(term.width), term.height // 2))
        space_below_input = 2
        reserved_h = 1 + input_h + 1 + space_below_input
        main_h = max(0, term.height - reserved_h)
        main_r = Region(0, 0, term.width, main_h)
        input_r = Region(0, main_r[1] + main_r[3] + 1, term.width, input_h)
        footer_r = Region(0, input_r[1] + input_r[3] + 1, term.width, space_below_input - 1)
    elif tui.sel_input:
        input_h = max(1, min(input_box.get_height(term.width - 2), term.height // 3))
        footer_r = Region(0, term.height - 2, term.width, 1)
        input_r = Region(0, footer_r[1] - input_h - 2, term.width, input_h + 2)
        main_r = Region(0, 0, term.width, input_r[1])
    else:
        main_r = Region(0, 0, term.width, term.height - 1)

    if tui.mode == "scroll":
        if inpt._keys:
            tui.mode = tui.prev_mode
            tui.app.stdout.write(term.enter_fullscreen)
            tui.app.stdout.flush()
            sys.stdout, sys.stderr = _sink, _sink
            buf.invalidate()
    elif tui.mode == "work":
        th = tui.app.theme
        tui.app.call("render_work_mode", tui, buf, inpt, main_r)
        div_color = th.invoking if tui.current.is_running() else th.muted
        buf.hline((0, main_r[1] + main_r[3], term.width, 1), txt_color=div_color)
        if not tui.current.ui_stack and not tui.ui_panel_stack:
            tui.app.call("render_work_mode_input", tui, buf, inpt, input_r, input_box)
        buf.hline((0, input_r[1] + input_r[3], term.width, 1), txt_color=div_color)
        render_workmodefooter_and_commands(tui, buf, footer_r, tui.current)
        if inpt.consume('KEY_ESCAPE'):
            tui.mode = "selection"
        if inpt.consume('KEY_CTRL_X') and tui.current.is_running():
            tui.current.stop_early = True
    elif tui.mode == "selection":
        th = tui.app.theme
        left, right = main_r.split_horizontal(1, 3)
        tui.app.call("render_selection_left", tui, buf, inpt, left)
        tui.app.call("render_selection_right", tui, buf, right)
        if tui.sel_input and not tui.current.ui_stack:
            buf.rect_line(input_r, txt_color=th.invoking if tui.current.is_running() else th.accent)
            buf.puts(input_r[0] + 2, input_r[1], f" {tui.current.name} ", txt_color=th.accent_alt)
            if input_box(buf, inpt, input_r.shrink(1), txt_color=th.text): tui.sel_input = False
            render_workmodefooter_and_commands(tui, buf, footer_r, tui.current)

    if tui.mode != "scroll":
        tui.app.call("render_status_line", tui, buf, inpt, status_r)
        if tui.ui_panel_stack:
            r = Region(3, 2, buf.w - 6, buf.h - 4)
            tui.ui_panel_stack[-1](buf, inpt, r)
        elif tui.current.ui_stack and (tui.mode == "work" or tui.sel_input):  # selection mode: space/h to answer
            r = Region(3, 2, buf.w - 6, buf.h - 4)
            tui.current.ui_stack[-1](buf, inpt, r)
            if not tui.current.ui_stack: tui.sel_input = False
        if tui.mode == "selection" and not tui.ui_panel_stack:  # leftover keys: bare ex6 stays navigable
            ctxs = tui.context_list()
            idx = ctxs.index(tui.current)
            if not tui.sel_input and inpt.consume('KEY_UP'): tui.current = ctxs[max(0, idx - 1)]
            if not tui.sel_input and inpt.consume('KEY_DOWN'): tui.current = ctxs[min(len(ctxs) - 1, idx + 1)]
            if inpt.consume('KEY_ENTER'): tui.mode, tui.sel_input = "work", False
        if tui.mode == "work":
            available = main_r[3] - 1
            if inpt.consume('KEY_PGUP'): tui.current._scroll_up += available // 2
            if inpt.consume('KEY_PGDOWN'): tui.current._scroll_up = max(0, tui.current._scroll_up - available // 2)
            if inpt.consume('KEY_UP'): tui.current._scroll_up += ARROW_SCROLL_SPEED
            if inpt.consume('KEY_DOWN'): tui.current._scroll_up = max(0, tui.current._scroll_up - ARROW_SCROLL_SPEED)
        if tui.mode != prev_mode:
            buf.invalidate()
        buf.flush(term, tui.app.stdout)



def _run_tui_loop(tui: TUI, real_stdout, real_stderr):
    term = tui.term

    try:
        with term.cbreak(), term.hidden_cursor(), term.fullscreen(), term.bracketed_paste():
            while True:
                _tui_loop(tui)
    except Exception:
        import traceback
        sys.stdout, sys.stderr = real_stdout, real_stderr
        try:
            real_stdout.write(term.exit_fullscreen)
            real_stdout.flush()
        except Exception:
            pass
        tb = traceback.format_exc()
        tui.app.debug_print(tb)
        print(tb)
        sys.exit(1)


if __name__ == "__main__":
    run_tui()

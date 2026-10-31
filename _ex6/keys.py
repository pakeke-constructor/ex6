# Dependencies: pyperclip
# Selection-mode keys. Work mode is for typing prompts; selection mode is for keys.
# Keys act on tui.current. Leading digits are a count (`3dd`). `?` lists everything.

import re
import pyperclip
import ex6
from ex6 import key
from _ex6.commands import _text_panel


def _idle(tui):
    ctx = tui.current
    if ctx.is_running():
        tui.echo(f"{ctx.name} is running (ctrl-x to stop)", error=True)
        return None
    return ctx


def _texts(ctx, role):
    msgs = [m for m in ctx.get_messages() if m.role == role]
    return [t for t in (ex6.tool_result_text(m.get_msg(ctx)) for m in msgs) if t.strip()]


def _nth_last(items, n):
    return items[-n] if 0 < n <= len(items) else None


def _yank(tui, text, what):
    if text is None: return tui.echo(f"no {what}", error=True)
    pyperclip.copy(text)
    n = text.count("\n") + 1
    tui.echo(f"yanked {what} ({n} line{'s' * (n > 1)})")


def _unique_name(tui, base):
    base = re.sub(r"\.\d+$", "", base)
    n = 2
    while tui.get_context(f"{base}.{n}"): n += 1
    return f"{base}.{n}"


def _add(tui, ctx):
    tui.add_context(ctx)
    tui.current = ctx
    return ctx


def _open_input(tui, text=None):
    if text is not None: tui.current.get_input_box().set_text(text)
    tui.sel_input = True


def _pop_prompts(tui, n):
    """Truncate to the Nth-last user prompt. Returns its text, or None."""
    ctx = _idle(tui)
    if not ctx: return None
    msgs = ctx.get_messages()
    users = [i for i, m in enumerate(msgs) if m.role == "user"]
    if not users:
        tui.echo("no prompts", error=True)
        return None
    cut = users[-min(n, len(users))]
    text = ex6.tool_result_text(msgs[cut].get_msg(ctx))
    ctx.truncate(cut)
    return text


# --- navigate / start ---

@key("j")
def down(tui, k):
    'Next agent.'
    ctxs = tui.context_list()
    tui.current = ctxs[min(len(ctxs) - 1, ctxs.index(tui.current) + k.count)]


@key("k")
def up(tui, k):
    'Prev agent.'
    ctxs = tui.context_list()
    tui.current = ctxs[max(0, ctxs.index(tui.current) - k.count)]


@key("<CR>")
def work(tui, k):
    'Open agent in work mode.'
    tui.mode = "work"


@key(" ")
def prompt(tui, k):
    'Type a prompt for the agent, without leaving the list.'
    _open_input(tui)


@key("/")
def command(tui, k):
    'Run a /command.'
    _open_input(tui, "/")


@key("h")
def human(tui, k):
    'Go to the nearest agent that needs a human (? in the list).'
    ctxs = tui.context_list()
    i = ctxs.index(tui.current)
    asks = sorted((c for c in ctxs if c.ui_stack), key=lambda c: abs(ctxs.index(c) - i))
    if not asks: return tui.echo("nobody needs you")
    tui.current = asks[0]
    tui.mode = "work"


@key("g")
def go(tui, k):
    'Go: run the agent without a new message.'
    if ctx := _idle(tui): ctx.invoke()


@key("z")
def yolo(tui, k):
    'Toggle yolo (auto-approve tools).'
    ctx = tui.current
    ctx.yolo = not ctx.yolo
    tui.echo(f"{ctx.name}: yolo {'ON' if ctx.yolo else 'off'}", error=ctx.yolo)


@key("<C-x>")
def stop(tui, k):
    'Stop the agent.'
    if tui.current.is_running(): tui.current.stop_early = True


@key("f")
def fork(tui, k):
    'Fork the agent (`f<space>` = fork and prompt).'
    ctx = tui.current
    _add(tui, ctx.fork(_unique_name(tui, ctx.name)))


@key("o")
def fork_prompt(tui, k):
    'Fork the agent and type a prompt for the fork.'
    fork(tui, k)
    _open_input(tui)


@key("n")
def new(tui, k):
    'New agent from the favorite (tui.set_favorite), type a prompt.'
    fav = tui.favorite or tui.current
    ctx = fav.fork(_unique_name(tui, fav.name))
    ctx.clear()
    _add(tui, ctx)
    _open_input(tui, "")


# --- yank ---

@key("yy")
def yank_prompt(tui, k):
    'Yank Nth-last user prompt.'
    _yank(tui, _nth_last(_texts(tui.current, "user"), k.count), "prompt")


@key("ya")
def yank_agent(tui, k):
    "Yank agent's Nth-last message."
    _yank(tui, _nth_last(_texts(tui.current, "assistant"), k.count), "message")


@key("yc")
def yank_code(tui, k):
    "Yank agent's Nth-last code block."
    blocks = [m.group(1).rstrip("\n") for t in _texts(tui.current, "assistant")
              for m in re.finditer(r"```[^\n]*\n(.*?)```", t, re.S)]
    _yank(tui, _nth_last(blocks, k.count), "code block")


@key("yl")
def yank_lines(tui, k):
    "Yank last N lines of agent's output."
    last = _nth_last(_texts(tui.current, "assistant"), 1)
    _yank(tui, "\n".join(last.strip().split("\n")[-k.count:]) if last else None, "lines")


# --- edit history ---

@key("p")
def pop(tui, k):
    'Pop the last N messages.'
    ctx = _idle(tui)
    if not ctx: return
    msgs = ctx.get_messages()
    keep = max(len(msgs) - k.count, sum(1 for m in msgs if m.role == "system"))
    ctx.truncate(keep)
    tui.echo(f"popped {len(msgs) - keep} message(s)")


@key("dd")
def delete_prompt(tui, k):
    'Delete last N user prompts (and everything after). Copies the prompt.'
    text = _pop_prompts(tui, k.count)
    if text is not None:
        pyperclip.copy(text)
        tui.echo(f"popped {k.count} prompt(s), copied")


@key("da")
def delete_spawned(tui, k):
    "Delete all agents ex6 didn't start with."
    spawned = [c for c in tui.contexts if c not in tui.boot_contexts]
    for c in spawned: tui.remove_context(c)
    tui.echo(f"deleted {len(spawned)} agent(s)")


@key("cc")
def change(tui, k):
    'Pop last user prompt, type a new one.'
    if _pop_prompts(tui, k.count) is not None: _open_input(tui, "")


@key("ca")
def change_append(tui, k):
    'Pop last user prompt, edit it.'
    text = _pop_prompts(tui, k.count)
    if text is not None: _open_input(tui, text)


# --- help ---

@key("?")
def show_help(tui, k):
    'Keys and commands.'
    _text_panel(tui, help_lines(tui.app))


def help_lines(app):
    lines = ["Keys  (selection mode; counts: 3j 2yy 3dd)", ""]
    for toks, fn in app.keys.items():
        name = "".join(toks).replace(" ", "<space>")
        lines.append(f"  {name:<10} {(fn.__doc__ or '').strip()}")
    lines += ["", "Commands  (/ in selection mode, or in work-mode input)", ""]
    for name, (fn, spec) in sorted(app.iter_commands()):
        args = " ".join(f"<{a}>" for a, _, _ in spec)
        lines.append(f"  /{name} {args}".ljust(24) + " ".join((fn.__doc__ or "").split()))
    lines += ["", "Type ;name in any prompt to paste _prompts/name.md (;;name = literal)."]
    return lines

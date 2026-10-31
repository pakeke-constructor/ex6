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
def down(tui, n):
    'Next agent.'
    ctxs = tui.context_list()
    tui.current = ctxs[min(len(ctxs) - 1, ctxs.index(tui.current) + n)]


@key("k")
def up(tui, n):
    'Prev agent.'
    ctxs = tui.context_list()
    tui.current = ctxs[max(0, ctxs.index(tui.current) - n)]


@key(" ", "a")
def prompt(tui, n):
    'Type a prompt for the agent (or answer its question), without leaving the list. Enter on empty = work mode.'
    _open_input(tui)


@key("/")
def command(tui, n):
    'Run a /command.'
    _open_input(tui, "/")


@key("h")
def human(tui, n):
    'Answer the nearest agent that needs a human (? in the list).'
    ctxs = tui.context_list()
    i = ctxs.index(tui.current)
    asks = sorted((c for c in ctxs if c.ui_stack), key=lambda c: abs(ctxs.index(c) - i))
    if not asks: return tui.echo("nobody needs you")
    tui.current = asks[0]
    _open_input(tui)


@key("g")
def go(tui, n):
    'Go: run the agent without a new message.'
    if ctx := _idle(tui): ctx.invoke()


@key("z")
def yolo(tui, n):
    'Toggle yolo (auto-approve tools).'
    ctx = tui.current
    ctx.yolo = not ctx.yolo
    tui.echo(f"{ctx.name}: yolo {'ON' if ctx.yolo else 'off'}", error=ctx.yolo)


@key("<C-x>")
def stop(tui, n):
    'Stop the agent.'
    if tui.current.is_running(): tui.current.stop_early = True


@key("f")
def fork(tui, n):
    'Fork the agent (`f<space>` = fork and prompt).'
    ctx = tui.current
    _add(tui, ctx.fork(_unique_name(tui, ctx.name)))


@key("o")
def fork_prompt(tui, n):
    'Fork the agent and type a prompt for the fork.'
    fork(tui, n)
    _open_input(tui)


@key("n")
def new(tui, n):
    'New agent from tui.favorite, type a prompt.'
    fav = tui.favorite or tui.current
    ctx = fav.fork(_unique_name(tui, fav.name))
    ctx.clear()
    _add(tui, ctx)
    _open_input(tui, "")


# --- yank ---

@key("yy")
def yank_prompt(tui, n):
    'Yank Nth-last user prompt.'
    _yank(tui, _nth_last(_texts(tui.current, "user"), n), "prompt")


@key("yr")
def yank_response(tui, n):
    "Yank agent's Nth-last response."
    _yank(tui, _nth_last(_texts(tui.current, "assistant"), n), "response")


@key("yc")
def yank_code(tui, n):
    "Yank agent's Nth-last code block."
    blocks = [m.group(1).rstrip("\n") for t in _texts(tui.current, "assistant")
              for m in re.finditer(r"```[^\n]*\n(.*?)```", t, re.S)]
    _yank(tui, _nth_last(blocks, n), "code block")


# --- edit history ---

@key("p")
def pop(tui, n):
    'Pop the last N messages.'
    ctx = _idle(tui)
    if not ctx: return
    msgs = ctx.get_messages()
    keep = max(len(msgs) - n, sum(1 for m in msgs if m.role == "system"))
    ctx.truncate(keep)
    tui.echo(f"popped {len(msgs) - keep} message(s)")


@key("dd")
def delete_prompt(tui, n):
    'Delete last N user prompts (and everything after). Copies the prompt.'
    text = _pop_prompts(tui, n)
    if text is not None:
        pyperclip.copy(text)
        tui.echo(f"popped {n} prompt(s), copied")


@key("da")
def delete_spawned(tui, n):
    "Delete all spawned agents (forks, subagents)."
    spawned = [c for c in tui.contexts if c.parent is not None]
    for c in spawned: tui.remove_context(c)
    tui.echo(f"deleted {len(spawned)} agent(s)")


@key("cc")
def change(tui, n):
    'Pop last user prompt, type a new one.'
    if _pop_prompts(tui, n) is not None: _open_input(tui, "")


@key("ca")
def change_append(tui, n):
    'Pop last user prompt, edit it.'
    text = _pop_prompts(tui, n)
    if text is not None: _open_input(tui, text)


# --- help ---

@key("?")
def show_help(tui, n):
    'Keys and commands.'
    _text_panel(tui, help_lines(tui.app))


@ex6.command
def help(tui):
    'List keys and commands.'
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
    lines += ["", ";name in a prompt pastes _prompts/name.md on Enter (shown first; ;;name = literal)."]
    return lines

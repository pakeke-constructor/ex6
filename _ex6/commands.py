# Dependencies: pyperclip
# Install: python -m pip install pyperclip


from typing import Optional
import subprocess
import threading
import json
import math
import textwrap
import pyperclip
import time
import ex6


@ex6.command
def clr(tui, name: Optional[str]):
    'Clear context messages.'
    ctx = tui.get_context(name) if name else tui.current
    if not ctx: return
    ctx.clear()


@ex6.command
def purge(tui):
    'Purge cached message content from current context.'
    ctx = tui.current
    if ctx: ctx.purge_cache()


@ex6.command
def pop(tui, n: Optional[int]):
    'Pop last N user messages and everything after each cutoff.'
    ctx = tui.current
    if not ctx or ctx.is_running(): return
    if n is None: n = 1
    if n <= 0: return
    user_idxs = [i for i, m in enumerate(ctx.get_messages()) if m.role == "user"]
    n = min(n, len(user_idxs))
    if n == 0: return
    ctx.truncate(user_idxs[-n])


@ex6.command
def yy(tui, n: Optional[int]):
    'Copy the Nth-last user prompt to clipboard.'
    ctx = tui.current
    if not ctx: return
    if n is None: n = 1
    if n <= 0: return
    user_messages = [m for m in ctx.get_messages() if m.role == "user"]
    if n > len(user_messages): return
    pyperclip.copy(user_messages[-n].get_msg(ctx))


@ex6.command
def del_context(tui, name: Optional[str]):
    'Delete a context.'
    ctx = tui.get_context(name) if name else tui.current
    if not ctx: return
    tui.remove_context(ctx)

del_context.__name__ = "del" # coz del is python keyword, and we want /del



@ex6.command
def fork(tui, name: Optional[str]):
    'Fork current context.'
    ctx = tui.current
    if not ctx: return
    tui.add_context(ctx.fork(name))


@ex6.command
def stop(tui):
    'Stop running LLM.'
    ctx = tui.current
    if ctx and ctx.is_running():
        ctx.stop_early = True


@ex6.command
def go(tui):
    'Start LLM without adding a user message.'
    ctx = tui.current
    if ctx and not ctx.is_running():
        ctx.invoke()


@ex6.command
def crash(tui):
    'Force a crash (debug).'
    raise RuntimeError("Crash!")


@ex6.command
def dmp(tui):
    'Dump current context to .EX6_CONTEXT_DUMP.txt.'
    ctx = tui.current
    if not ctx: return
    with open(".EX6_CONTEXT_DUMP.txt", "w", encoding="utf-8") as f:
        f.write(format_context(ctx))


def format_context(ctx) -> str:
    lines = [f"=== Context: {ctx.name} ({ctx.model}) ===\n"]
    for i, msg in enumerate(ctx.get_messages()):
        label = msg.role
        if msg.tool_call_id: label += f" (tool_call_id={msg.tool_call_id})"
        if msg.attachments: label += f" ({len(msg.attachments)} attachment(s))"
        lines.append(f"--- [{i}] {label} ---")
        lines.append(ex6.tool_result_text(msg.get_msg(ctx)))
        for tc in msg.tool_calls or []:
            name = tc['name']
            args = tc.get("args", {})
            if not args:
                lines.append(f"  <{name} />")
                continue
            lines.append(f"  <{name}>")
            for k, v in args.items():
                v_str = str(v)
                if '\n' in v_str:
                    lines.append(f"    {k} =")
                    for line in v_str.split('\n'):
                        lines.append(f"      {line}")
                else:
                    lines.append(f"    {k} = {v_str}")
            lines.append(f"  </{name}>")
        for nc in msg.native_calls or []:
            lines.append(f"  <native:{nc['name']} args={json.dumps(nc['args'])}>")
            for line in str(nc['result']).split('\n'):
                lines.append(f"    {line}")
            lines.append(f"  </native:{nc['name']}>")
        lines.append("")
    return "\n".join(lines)


@ex6.command
def dbg_keys(tui):
    'Toggle key debug logging.'
    tui.log_keys = not tui.log_keys
    tui.app.debug_print(f"Key logging {'ON' if tui.log_keys else 'OFF'}")


def _llm_one_shot(app, model: str, system: str, user: str) -> str:
    """Synchronously run one LLM call. Returns assistant text."""
    ctx = ex6.Context(app, name="__tmp_cm__", model=model, reasoning="none")
    ctx.append_message(ex6.Message(role="system", content=system))
    ctx.append_message(ex6.Message(role="user", content=user))
    result_text = []
    for item in ctx.get_invoker()(ctx):
        if isinstance(item, ex6.ResponseChunk) and item.type == "text":
            result_text.append(item.content)
    return "".join(result_text).strip()



CM_SYSTEM_PROMPT = """
You write one-line git commit messages.
You MUST use the "Conventional Commits" specification.

<type>[optional scope]: description

Structure examples:
feat(...) ...
fix(...) ...
docs: ...
chore(...) ...
perf: ...
ci: ...
refactor(...) ...

Key strategies:
- If it's a small, tiny change and is unclear what the purpose is: Do NOT guess. Just say `chore(file)` or `fix(file)`, with nothing else. Do NOT guess.
- If many changes to existing systems, but the API / user facing code remains the same, it's likely a refactor
- If there's a CLEAR improvement, i.e. an new API to use, or new feature users will see, it's likely a feature, (feat)

One line only. No quotes. No explanation.
Be extremely concise, grammatical correctness is not important.
"""


def _text_panel(tui, lines):
    """Push a scrollable text panel. ESC to close."""
    scroll = [0]
    def draw(buf, inpt, r):
        x, y, w, h = r
        th = tui.app.theme
        buf.fill(r, ' ')
        buf.rect_line(r, txt_color=th.accent)
        if inpt.consume('KEY_UP') and scroll[0] > 0: scroll[0] -= 1
        if inpt.consume('KEY_DOWN'): scroll[0] += 1
        visible = h - 2
        max_scroll = max(0, len(lines) - visible)
        if scroll[0] > max_scroll: scroll[0] = max_scroll
        for i, line in enumerate(lines[scroll[0]:scroll[0] + visible]):
            buf.puts(x + 2, y + 1 + i, line[:w - 4], txt_color=th.text)
    tui.ui_panel_stack.append(draw)


SMP = r'''
Take a step back, and check for a simpler solution.
If lot of code was added/changed, take a step back and evaluate the actual problem.
If the new code seems hacky, look at callers/users of the system, and reason about the intention of the system; maybe something else can change, or the requirements can be relaxed.
Otherwise, if the code is clean and minimal; that's fine, carry on.
'''

@ex6.command
def smp(tui, additional_msg: Optional[str]):
    'Simplify command. Invokes agent, asking it to attempt to simplify or shorten recent code'
    ctx = tui.current
    if not ctx: return
    msg = SMP
    if additional_msg:
        msg += "\n\nAdditional user note:" + additional_msg
    ctx.invoke(msg)



SSOT_CHECK = r'''
Evaluate the statefulness of the system/code you just worked on.
(Bad state is one of the most common causes of bugs, and we want to avoid it.)
Some guidelines, in order:
- If it's possible to remove the state entirely via smarter code: REMOVE IT.
- Otherwise, if the state can't be removed, try make it a single-source-of-truth (SSOT).
- Otherwise, if state must be duplicated, then make sure the state is either short-lived or recomputed frequently.
- Lastly, if the bad state can't be short-lived or recomputed, think about a way to invalidate it, or make the consumers aware of the duplication.
Don't forget the bigger picture. 
If the system/code is clean and minimal; that's fine, no changes needed.

(Rule of thumb: If a goblin comes in and changes a bunch of memory/state randomly;
the program should just change behaviour; nothing should ever become broken.)
'''

@ex6.command
def ssot(tui, additional_msg: Optional[str]):
    'State-check command. Invokes agent, asking it to attempt to remove fragile state'
    ctx = tui.current
    if not ctx: return
    msg = SSOT_CHECK
    if additional_msg:
        msg += "\n\nAdditional user note:" + additional_msg
    ctx.invoke(msg)



SHORT = r'''
Review the code you just worked on, and remove redundant code.
Look for: dead/unused code, duplicated logic, unnecessary checks or error handling, needless variables/indirection, and comments that restate the code.
Prefer concise forms, but keep it readable; no clever one-liners.
Don't change behaviour too much.
If the code is already short and clean; that's fine, no changes needed.
'''

@ex6.command
def short(tui, additional_msg: Optional[str]):
    'Shorten command. Invokes agent, asking it to remove redundant code'
    ctx = tui.current
    if not ctx: return
    msg = SHORT
    if additional_msg:
        msg += "\n\nAdditional user note:" + additional_msg
    ctx.invoke(msg)





_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def _status_panel(tui, title, status):
    """Push an animated status card with a fake progress bar. Returns (state, finish); auto-closes on success."""
    s = {"status": status, "lines": [], "ok": None, "t0": time.time(), "done_at": 0.0}

    def draw(buf, inpt, r):
        th = tui.app.theme
        now = time.time()
        color = {True: th.success, False: th.error}.get(s["ok"], th.accent)
        icon = {True: "✓", False: "✗"}.get(s["ok"], _SPINNER[int(now * 12) % len(_SPINNER)])
        frac = 1.0 if s["ok"] is not None else min(0.95, 1 - math.exp(-(now - s["t0"]) / 3))  # fake
        w = min(r[2], 72)
        lines = [l for text in s["lines"] for l in textwrap.wrap(text, w - 6) or [""]]
        h = min(r[3], len(lines) + 7)
        x, y = r[0] + (r[2] - w) // 2, r[1] + (r[3] - h) // 3
        buf.fill((x, y, w, h), ' ')
        buf.rect_line((x, y, w, h), txt_color=color)
        buf.puts(x + 2, y, f" {title} ", txt_color=th.accent_alt)
        buf.puts(x + 3, y + 2, f"{icon} {s['status']}", txt_color=color)
        bar_w = w - 12
        full = int(frac * bar_w)
        buf.puts(x + 3, y + 3, "█" * full, txt_color=color)
        buf.puts(x + 3 + full, y + 3, "─" * (bar_w - full) + f"{int(frac * 100):>4}%", txt_color=th.muted)
        for i, line in enumerate(lines[:h - 7]):
            buf.puts(x + 3, y + 5 + i, line, txt_color=th.text)
        if s["ok"] and now - s["done_at"] > 0.7 and draw in tui.ui_panel_stack:
            tui.ui_panel_stack.remove(draw)

    def finish(ok, status, *lines):
        s["lines"] += lines
        s.update(ok=ok, status=status, done_at=time.time())

    tui.ui_panel_stack.append(draw)
    return s, finish


@ex6.command
def cm(tui, msg: Optional[str]):
    """Generate a commit message from git diff and commit."""
    s, finish = _status_panel(tui, "git commit", "Reading diff")

    def run():
        _git("add", ".")
        diff = _git("diff", "-M", "-D", "HEAD").stdout
        if not diff:
            return finish(True, "Nothing to commit")
        s["status"] = "Writing commit message"
        stat = _git("diff", "--stat", "-M", "HEAD").stdout
        hint = f"\nUser hint: {msg}" if msg else ""
        user = f"Write a commit message for this diff:{hint}\n\n{stat}\n{diff[:8000]}"
        from _ex6.models import M
        commit_msg = _llm_one_shot(tui.app, M.GPT_LUNA_LATEST.id, CM_SYSTEM_PROMPT, user)
        s["lines"].append(commit_msg)
        s["status"] = "Committing"
        result = _git("commit", "-m", commit_msg)
        if result.returncode == 0:
            finish(True, "Committed")
        else:
            finish(False, "git commit failed", *(result.stderr + result.stdout).strip().split("\n"))

    threading.Thread(target=run, daemon=True).start()


@ex6.command
def sync(tui):
    """Fetch origin, then merge origin/<branch>, then origin/main or origin/master."""
    s, finish = _status_panel(tui, "git sync", "Fetching origin")

    def run():
        fetch = _git("fetch", "origin")
        if fetch.returncode != 0:
            return finish(False, "git fetch failed", *fetch.stderr.strip().split("\n"))
        branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        for target in dict.fromkeys([f"origin/{branch}", "origin/main", "origin/master"]):
            s["status"] = f"Merging {target}"
            merge = _git("merge", target)
            out = (merge.stdout + merge.stderr).strip()
            if merge.returncode == 0:
                s["lines"].append(f"{target}: {out.splitlines()[-1].strip() if out else 'merged'}")
            elif "not something we can merge" not in out:
                return finish(False, f"Merging {target} failed", *out.split("\n"))
        finish(True, "Synced" if s["lines"] else "No remote branches found")

    threading.Thread(target=run, daemon=True).start()


@ex6.command
def help(tui):
    lines = ["Commands:"]
    for name, (fn, spec) in sorted(tui.app.iter_commands()):
        args = " ".join(f"<{a}>" for a, _, _ in spec)
        doc = " ".join((fn.__doc__ or "").split())
        line = f"  /{name} {args}".rstrip()
        lines.append(f"{line}  {doc}" if doc else line)
    _text_panel(tui, lines)



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
_BAR_PARTIAL = " ▏▎▍▌▋▊▉"

CM_STAGES = [  # (label, bar fraction at end of stage, fake time constant in seconds)
    ("Staging changes", 0.08, 0.3),
    ("Reading diff", 0.15, 0.3),
    ("Writing commit message", 0.9, 3.0),
    ("Committing", 1.0, 0.5),
]


@ex6.command
def cm(tui, msg: Optional[str]):
    """Generate a commit message from git diff and commit."""
    # result: None (running) | "ok" | "empty" | "fail"
    s = {"stage": 0, "t0": time.time(), "times": [], "result": None, "done_at": 0.0,
         "shown": 0.0, "last": time.time(), "commit": "", "stat": "", "notes": [], "errors": []}

    def draw(buf, inpt, r):
        th = tui.app.theme
        now = time.time()
        stage, result = s["stage"], s["result"]

        # Fake progress: ease toward the current stage's end, then smooth the displayed value.
        if result in ("ok", "empty"):
            target = 1.0
        elif result == "fail":
            target = s["shown"]
        else:
            i = min(stage, len(CM_STAGES) - 1)
            a = CM_STAGES[i - 1][1] if i else 0.0
            _, b, tau = CM_STAGES[i]
            target = a + (b - a) * (1 - math.exp(-(now - s["t0"]) / tau))
        s["shown"] += (target - s["shown"]) * min(1.0, (now - s["last"]) * 10)
        s["last"] = now

        pw = min(r[2], 72)
        inner = pw - 6
        rows = [[]]
        for i, (label, _, _) in enumerate(CM_STAGES):
            t = ""
            if i < stage:
                icon, ic, lc, t = "✓", th.success, th.text, f"{s['times'][i]:.1f}s"
            elif i == stage and result == "fail":
                icon, ic, lc = "✗", th.error, th.error
            elif i == stage and result is None:
                icon, ic, lc = _SPINNER[int(now * 12) % len(_SPINNER)], th.running, th.text
                t = f"{now - s['t0']:.1f}s"
            elif result == "empty":
                icon, ic, lc = "-", th.muted, th.muted
            else:
                icon, ic, lc = "·", th.muted, th.muted
            gap = " " * max(1, inner - 2 - len(label) - len(t))
            rows.append([(icon + " ", ic), (label, lc), (gap + t, th.muted)])
        rows.append([])

        bar_color = {"ok": th.success, "empty": th.warning, "fail": th.error}.get(result, th.accent)
        bar_w = inner - 5
        cells = s["shown"] * bar_w
        full = int(cells)
        part = _BAR_PARTIAL[int((cells - full) * 8)].strip() if full < bar_w else ""
        filled = "█" * full + part
        rows.append([(filled, bar_color), ("─" * (bar_w - len(filled)), th.muted),
                     (f"{int(s['shown'] * 100):>4}%", th.text)])

        for note in s["notes"]:
            rows.append([(note[:inner], th.text_alt)])
        if s["commit"]:
            rows.append([])
            for line in textwrap.wrap(s["commit"], inner) or [""]:
                rows.append([(line, th.md_bold)])
        if s["stat"]:
            rows.append([(s["stat"][:inner], th.muted)])
        if result == "empty":
            rows += [[], [("Nothing to commit.", th.warning)]]
        if result == "ok":
            rows += [[], [("Committed.", th.success)]]
        if s["errors"]:
            rows.append([])
        for line in s["errors"]:
            rows.append([(line[:inner], th.error)])
        rows.append([])

        ph = min(r[3], len(rows) + 2)
        panel = ex6.Region(r[0] + (r[2] - pw) // 2, r[1] + max(0, (r[3] - ph) // 3), pw, ph)
        px, py, _, _ = panel
        buf.fill(panel, ' ')
        buf.rect_line(panel, txt_color=bar_color)
        buf.puts(px + 2, py, " git commit ", txt_color=th.accent_alt)
        if result == "fail":
            buf.puts(px + pw - 16, py + ph - 1, " esc to close ", txt_color=th.muted)
        for j, row in enumerate(rows[:ph - 2]):
            cx = px + 3
            for text, color in row:
                buf.puts(cx, py + 1 + j, text, txt_color=color)
                cx += len(text)

        if result in ("ok", "empty") and now - s["done_at"] >= 1.5 and draw in tui.ui_panel_stack:
            tui.ui_panel_stack.remove(draw)

    tui.ui_panel_stack.append(draw)

    def advance():
        s["times"].append(time.time() - s["t0"])
        s["stage"] += 1
        s["t0"] = time.time()

    def finish(result):
        s["result"] = result
        s["done_at"] = time.time()

    def run():
        subprocess.run(["git", "add", "."], capture_output=True)
        advance()
        diff = subprocess.run(["git", "diff", "-M", "-D", "HEAD"], capture_output=True, text=True).stdout
        if not diff:
            finish("empty")
            return
        s["stat"] = subprocess.run(["git", "diff", "--shortstat", "-M", "HEAD"], capture_output=True, text=True).stdout.strip()

        from _ex6.models import M
        model = M.GPT_LUNA_LATEST.id

        hint = f"User hint: {msg}" if msg else ""
        diff_for_llm = diff
        if len(diff) > 8000:
            s["notes"].append("Large diff; using git diff --stat summary.")
            diff_for_llm = subprocess.run(
                ["git", "diff", "--stat", "-M", "HEAD"], capture_output=True, text=True
            ).stdout
            if len(diff_for_llm) > 8000:
                diff_for_llm = diff_for_llm[:8000] + "\n[Summary truncated.]"
            diff_for_llm = "[Full diff omitted; file change summary only.]\n" + diff_for_llm
        advance()
        system = CM_SYSTEM_PROMPT
        user = f"Write a commit message for this diff:{hint}\n\n{diff_for_llm}"
        try:
            s["commit"] = _llm_one_shot(tui.app, model, system, user)
        except Exception as e:
            s["errors"] = [f"LLM call failed: {e}"]
            finish("fail")
            return
        advance()

        subprocess.run(["git", "add", "."], capture_output=True)
        result = subprocess.run(["git", "commit", "-m", s["commit"]], capture_output=True, text=True)
        if result.returncode == 0:
            advance()
            finish("ok")
        else:
            out = (result.stderr + "\n" + result.stdout).strip()
            s["errors"] = ["git commit failed:"] + out.split('\n')
            finish("fail")

    threading.Thread(target=run, daemon=True).start()

@ex6.command
def sync(tui):
    """Fetch origin, then merge origin/<branch>, then origin/main or origin/master."""
    output_lines = ["Fetching origin..."]
    done = [False]

    def mark_done():
        if not done[0]:
            output_lines.append("Press any key to close.")
            done[0] = True

    def draw(buf, inpt, r):
        x, y, w, h = r
        th = tui.app.theme
        content_h = max(6, len(output_lines) + 2)
        panel = ex6.Region(x, y, w, min(content_h, h))
        px, py, pw, ph = panel
        buf.fill(panel, ' ')
        buf.rect_line(panel, txt_color=th.accent)
        visible = ph - 2
        start = max(0, len(output_lines) - visible)
        for i, line in enumerate(output_lines[start:start + visible]):
            buf.puts(px + 2, py + 1 + i, line[:pw - 4], txt_color=th.text)
        if done[0] and inpt._keys:
            inpt._keys.clear()
            tui.ui_panel_stack.pop()

    tui.ui_panel_stack.append(draw)

    def run():
        fetch = subprocess.run(["git", "fetch", "origin"], capture_output=True, text=True)
        if fetch.returncode != 0:
            output_lines.append("git fetch failed:")
            if fetch.stderr:
                output_lines.extend(fetch.stderr.split('\n'))
            mark_done()
            return

        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
        name = branch.stdout.strip() if branch.returncode == 0 else ""

        targets = []
        if name and name != "HEAD":
            targets.append(f"origin/{name}")
        targets.extend(["origin/main", "origin/master"])
        targets = list(dict.fromkeys(targets))

        merged_any = False
        for target in targets:
            output_lines.append(f"Merging {target}...")
            merge = subprocess.run(["git", "merge", target], capture_output=True, text=True)
            if merge.returncode == 0:
                merged_any = True
                output_lines.append(f"Merged {target}.")
                if merge.stdout:
                    output_lines.extend(merge.stdout.split('\n'))
                if merge.stderr:
                    output_lines.extend(merge.stderr.split('\n'))
                continue

            merge_text = (merge.stdout or "") + "\n" + (merge.stderr or "")
            if "not something we can merge" in merge_text:
                output_lines.append(f"Skipping {target} (missing remote branch).")
                continue

            output_lines.append("git merge failed:")
            if merge.stdout:
                output_lines.extend(merge.stdout.split('\n'))
            if merge.stderr:
                output_lines.extend(merge.stderr.split('\n'))
            mark_done()
            return

        if not merged_any:
            output_lines.append("No matching remote branches found.")

        mark_done()

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



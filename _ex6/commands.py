
from typing import Optional
import subprocess
import threading
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
def yolo(tui):
    'Toggle auto-approve tools.'
    ctx = tui.current
    if not ctx: return
    ctx.yolo = not ctx.yolo


@ex6.command
def crash(tui):
    'Force a crash (debug).'
    raise RuntimeError("Crash!")


def _llm_one_shot(app, model: str, system: str, user: str) -> str:
    """Synchronously run one LLM call. Returns assistant text."""
    ctx = ex6.Context(app, name="__tmp_cm__", model=model, reasoning="none")
    ctx.append_message(ex6.Message(role="system", content=system))
    ctx.append_message(ex6.Message(role="user", content=user))
    result_text = []
    for item in app.get_implementation("invoke_llm")(ctx):
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
- If small one-line change and is unclear what the purpose is, it's likely a fix
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
    'Simplify command. Invokes agent, asking it to attempt to simpllfy or shorten recent code'
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





@ex6.command
def cm(tui, msg: Optional[str]):
    """Generate a commit message from git diff and commit."""
    output_lines = ["Generating commit message..."]

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
    done_time = [None]

    def draw_auto_close(buf, inpt, r):
        draw(buf, inpt, r)
        if done_time[0] is not None and time.time() - done_time[0] >= 0.5:
            tui.ui_panel_stack.pop()

    tui.ui_panel_stack.append(draw_auto_close)

    def run():
        subprocess.run(["git", "add", "."], capture_output=True)
        diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True).stdout
        if not diff:
            output_lines.append("No changes to commit.")
            done_time[0] = time.time()
            return

        from _ex6.models import M
        model = M.GEMINI31_FLASH_LITE.id

        hint = f"User hint: {msg}" if msg else ""
        diff_for_llm = diff
        if len(diff) > 8000:
            output_lines.append("Large diff; using git diff --stat summary.")
            diff_for_llm = subprocess.run(
                ["git", "diff", "--stat", "HEAD"], capture_output=True, text=True
            ).stdout
            if len(diff_for_llm) > 8000:
                diff_for_llm = diff_for_llm[:8000] + "\n[Summary truncated.]"
            diff_for_llm = "[Full diff omitted; file change summary only.]\n" + diff_for_llm
        system = CM_SYSTEM_PROMPT
        user = f"Write a commit message for this diff:{hint}\n\n{diff_for_llm}"
        commit_msg = _llm_one_shot(tui.app, model, system, user)
        output_lines.append(f"Commit: {commit_msg}")

        subprocess.run(["git", "add", "."], capture_output=True)
        result = subprocess.run(["git", "commit", "-m", commit_msg], capture_output=True, text=True)
        if result.returncode == 0:
            output_lines.append("Committed.")
        else:
            output_lines.append(f"git commit failed:")
            output_lines.extend(result.stderr.split('\n'))
        done_time[0] = time.time()

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
        args = " ".join(f"<{a}>" for a, _ in spec)
        doc = (fn.__doc__ or "").strip()
        line = f"  /{name} {args}".rstrip()
        lines.append(f"{line}  {doc}" if doc else line)
    _text_panel(tui, lines)



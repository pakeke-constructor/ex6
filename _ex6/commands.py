
from typing import Optional
import subprocess
import threading
import time
import ex6


@ex6.command
def clr(name: Optional[str]):
    'Clear context messages.'
    ctx = ex6.state.contexts.get(name) if name else ex6.state.current
    if not ctx: return
    ctx.clear()


@ex6.command
def delete(name: Optional[str]):
    'Delete a context.'
    ctx = ex6.state.contexts.get(name) if name else ex6.state.current
    if not ctx: return
    del ex6.state.contexts[ctx.name]
    if ex6.state.current is ctx:
        ex6.state.current = None


@ex6.command
def fork(name: Optional[str]):
    'Fork current context.'
    ctx = ex6.state.current
    if not ctx: return
    ctx.fork(name)


@ex6.command
def stop():
    'Stop running LLM.'
    ctx = ex6.state.current
    if ctx and ctx.is_running():
        ctx.stop_early = True


@ex6.command
def yolo():
    'Toggle auto-approve tools.'
    ctx = ex6.state.current
    if not ctx: return
    ctx.yolo = not ctx.yolo


@ex6.command
def crash():
    'Force a crash (debug).'
    raise RuntimeError("Crash!")


def _llm_one_shot(model: str, system: str, user: str) -> str:
    """Synchronously run one LLM call. Returns assistant text."""
    ctx = ex6.Context(name="__tmp_cm__", model=model, reasoning="none")
    ctx.messages.append(ex6.Message(role="system", content=system))
    ctx.messages.append(ex6.Message(role="user", content=user))
    result_text = []
    for item in ex6.invoke_llm(ctx):
        if isinstance(item, ex6.ResponseChunk) and item.type == "text":
            result_text.append(item.content)
    del ex6.state.contexts["__tmp_cm__"]
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

One line only. No quotes. No explanation.
Be extremely concise, grammatical correctness is not important.
"""


def _text_panel(lines):
    """Push a scrollable text panel. ESC to close."""
    scroll = [0]
    def draw(buf, inpt, r):
        x, y, w, h = r
        th = ex6.state.theme
        buf.fill(r, ' ')
        buf.rect_line(r, txt_color=th.accent)
        if inpt.consume('KEY_UP') and scroll[0] > 0: scroll[0] -= 1
        if inpt.consume('KEY_DOWN'): scroll[0] += 1
        visible = h - 2
        max_scroll = max(0, len(lines) - visible)
        if scroll[0] > max_scroll: scroll[0] = max_scroll
        for i, line in enumerate(lines[scroll[0]:scroll[0] + visible]):
            buf.puts(x + 2, y + 1 + i, line[:w - 4], txt_color=th.text)
    ex6.push_ui_panel(draw)



CONDENSE_MSG = r"""Your context window is getting large. You MUST condense now.
Current token usage: {tokens} tokens {estimate_note}.

Call `condense` or `compact` to collapse your context.
Summarize ALL important findings, decisions, and file locations. Keep any files you'll need to edit soon.
"""

@ex6.command
def c(additional_msg: Optional[str]):
    'Invokes agent, asking it to compact/condense itself.'
    ctx = ex6.state.current
    if not ctx: return
    tokens = ctx.token_count()
    estimate_note = " (estimated)" if ctx.is_token_count_estimate() else ""
    msg = CONDENSE_MSG.format(tokens=tokens, estimate_note=estimate_note)
    if additional_msg:
        msg += "\nAdditional user note: " + additional_msg
    ctx.invoke(msg)


SMP = r'''
Take step back, and check for a simpler solution.
If lot of code was added/changed, take a step back and evaluate the actual problem.
If the new code seems hacky, look at callers/users of the system, and reason about the intention of the system; maybe something else can change, or the requirements can be relaxed.
Otherwise, if the code is clean and minimal; that's fine, carry on.
'''

@ex6.command
def smp(additional_msg: Optional[str]):
    'Invokes agent, asking it to attempt to simpllfy or shorten recent code'
    ctx = ex6.state.current
    if not ctx: return
    msg = SMP
    if additional_msg:
        msg += "\n\nAdditional user note:" + additional_msg
    ctx.invoke(msg)


@ex6.command
def cm(msg: Optional[str]):
    """Generate a commit message from git diff and commit."""
    output_lines = ["Generating commit message..."]

    def draw(buf, inpt, r):
        x, y, w, h = r
        th = ex6.state.theme
        buf.fill(r, ' ')
        buf.rect_line(r, txt_color=th.accent)
        visible = h - 2
        start = max(0, len(output_lines) - visible)
        for i, line in enumerate(output_lines[start:start + visible]):
            buf.puts(x + 2, y + 1 + i, line[:w - 4], txt_color=th.text)
    done_time = [None]

    def draw_auto_close(buf, inpt, r):
        draw(buf, inpt, r)
        if done_time[0] is not None and time.time() - done_time[0] >= 0.5:
            ex6.pop_ui_panel()

    ex6.push_ui_panel(draw_auto_close)

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
        system = CM_SYSTEM_PROMPT
        user = f"Write a commit message for this diff:{hint}\n\n{diff[:8000]}"

        commit_msg = _llm_one_shot(model, system, user)
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
def help():
    lines = ["Commands:"]
    for name, (fn, spec) in sorted(ex6._commands.items()):
        args = " ".join(f"<{a}>" for a, _ in spec)
        doc = (fn.__doc__ or "").strip()
        line = f"  /{name} {args}".rstrip()
        lines.append(f"{line}  {doc}" if doc else line)
    _text_panel(lines)


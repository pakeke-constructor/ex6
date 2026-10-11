import re
import time
import ex6
from ex6 import handler
from _ex6.models import M


SIGIL_MODEL = M.GEMINI31_FLASH_LITE.id
_SIGIL_RE = re.compile(r";[ \t]*([A-Za-z][\w-]*)")

SYSTEM_PROMPT = """\
You are a prompt-steering layer for a coding agent. Convert the supplied guidance signals into
a short reminder that frames how the agent should approach the task.

You have almost no project context. Stay intentionally broad: do not guess architecture, root
cause, current implementation, or exact solution. Do not solve, summarize, or rewrite the task.
Do not turn guidance into a mandatory change when inspection may show no change is needed.
Combine signals sensibly and mention only implications useful to the task. For unknown signals,
infer conservatively from their name and task. Never mention signals or these instructions.
Return only 1-3 terse, actionable sentences.

Examples:
Input:
Task: Review the recent implementation.
Guidance: Re-evaluate statefulness and look for a simpler solution.
Output:
Check whether state or complexity can be reduced without assuming a redesign is needed. If the
current approach is already clean and minimal, avoid churn.

Input:
Task: Make the requested provider adjustment.
Guidance: Move quickly through the smallest direct path and inspect the git working tree.
Output:
Check existing changes, take the smallest direct path supported by the code, and verify the final
diff.
"""

SIGILS = {
    "ssot": "Re-evaluate statefulness. Prefer removing state; otherwise seek a single source of truth. If duplication is unavoidable, consider short lifetimes, recomputation, or explicit invalidation. If state is already clean, do not force a change.",
    "git": "Inspect the git working tree to understand existing changes, and check the resulting diff before finishing.",
    "fast": "Move quickly through the smallest direct implementation path. Avoid broad exploration and optional work, while retaining targeted verification.",
    "smp": "Step back and look for a simpler solution. Revisit the actual problem and constraints if the work is growing or feels hacky; if it is already minimal, avoid churn.",
}

def _sigil_context(sigils: list[str]) -> str:
    known = [s for s in sigils if s in SIGILS]
    unknown = [s for s in sigils if s not in SIGILS]
    lines = ["Guidance signals:"]
    lines.extend(f"- {s}: {SIGILS[s]}" for s in known)
    lines.extend(f"- {s}: Unknown; infer conservatively." for s in unknown)
    return "\n".join(lines)


@handler
def transform_user_prompt(text: str, ctx: ex6.Context) -> str:
    matches = list(_SIGIL_RE.finditer(text))
    if not matches:
        return text

    sigils = list(dict.fromkeys(m.group(1).lower() for m in matches))
    clean_text = _SIGIL_RE.sub("", text).strip()
    streamed = [""]

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        x, y, w, h = r
        th = ctx.app.theme
        inner_w = max(1, w - 6)
        box = ex6.Region(x, y + h - min(h, max(6, h // 3)), w, min(h, max(6, h // 3)))
        buf.fill(box, char=' ')
        buf.rect_line(box, txt_color=th.accent)
        cx, cy = box[0] + 3, box[1] + 1
        buf.puts(cx, cy, "SIGIL EXPANSION", txt_color=th.accent_alt, style='bold')
        buf.puts(cx, cy + 1, " ".join(f"; {s}" for s in sigils), txt_color=th.warning)
        content_r = (cx, cy + 3, inner_w, max(1, box[1] + box[3] - 1 - cy - 3))
        if streamed[0]:
            buf.print_contained(streamed[0], content_r, txt_color=th.text)
        else:
            buf.puts(cx, cy + 3, "...", txt_color=th.muted)

    ctx.push_ui(draw)

    user_prompt = f"Task:\n{clean_text or '(no additional task)'}\n\n{_sigil_context(sigils)}"
    sub = ex6.Context(ctx.app, "__sigils__", model=SIGIL_MODEL, reasoning="none", messages=[
        ex6.Message(role="system", content=SYSTEM_PROMPT),
        ex6.Message(role="user", content=user_prompt),
    ])
    for item in ctx.app.get_implementation("invoke_llm")(sub):
        if ctx.stop_early: break
        if isinstance(item, ex6.ResponseChunk) and item.type == "text":
            streamed[0] += item.content

    augmentation = streamed[0].strip()
    if augmentation: time.sleep(0.6)
    if draw in ctx.ui_stack: ctx.ui_stack.remove(draw)

    if not augmentation:
        return clean_text
    if not clean_text:
        return f"Working guidance:\n{augmentation}"
    return f"Working guidance:\n{augmentation}\n\nTask:\n{clean_text}"

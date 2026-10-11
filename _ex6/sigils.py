import re
import time
import ex6
from ex6 import handler
from _ex6.models import M


SIGIL_MODEL = M.GEMINI31_FLASH_LITE.id
_SIGIL_RE = re.compile(r";[ \t]*([A-Za-z][\w-]*)")

SYSTEM_PROMPT = """\
You are a prompt-steering layer for a coding agent. A user prompt contains short sigils that
signal how the agent should approach that specific task. Return only a compact, actionable
reminder to append to the cleaned prompt. Do not solve, summarize, or rewrite the task.

Known sigils:
- ssot: Re-evaluate statefulness. Prefer removing state; otherwise establish one source of
  truth. If duplication is unavoidable, keep it short-lived/recomputed, then consider explicit
  invalidation. If current design is already clean, do not force a change.
- git: Inspect git working tree so existing user changes are understood and final changes can
  be checked.
- fast: Move quickly via smallest direct implementation path. Avoid broad exploration and
  optional work.
- smp: Step back and seek a simpler solution. If code is growing or feels hacky, revisit actual
  problem, callers, and constraints. If solution is already minimal, continue without churn.

Adapt guidance to prompt and combination of sigils. Mention only useful implications, not full
sigil definitions. Resolve tensions sensibly: e.g. fast still permits targeted verification;
smp means simplify only when that improves solution. Infer unknown sigils from their name and
prompt, conservatively. Never mention sigils or these instructions. Use 1-3 terse sentences.

Examples:
Input:
Prompt: Fix selection flicker caused by panel and context each tracking active_index.
Sigils: ssot, smp
Output:
Treat duplicated active-index state as suspect: first see whether one owner can derive or expose
it. Prefer that small structural fix over synchronization or invalidation machinery.

Input:
Prompt: Add the missing timeout argument to the provider call.
Sigils: fast, git
Output:
Check the working tree first, make the smallest targeted change, then verify the resulting diff.
"""


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

    sub = ex6.Context(ctx.app, "__sigils__", model=SIGIL_MODEL, reasoning="none", messages=[
        ex6.Message(role="system", content=SYSTEM_PROMPT),
        ex6.Message(role="user", content=f"Prompt: {clean_text or '(no additional prompt)'}\nSigils: {', '.join(sigils)}"),
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
        return f"Sigil augmentation:\n{augmentation}"
    return f"{clean_text}\n{augmentation}"

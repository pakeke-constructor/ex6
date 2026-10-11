import re
import time
import ex6
from ex6 import handler
from _ex6.models import M


SIGIL_MODEL = M.GEMINI31_FLASH_LITE.id
_SIGILS = ("ssot", "git", "fast", "smp")
_SIGIL_RE = re.compile(rf";[ \t]*({'|'.join(_SIGILS)})(?![\w-])", re.IGNORECASE)

SYSTEM_PROMPT = """\
You turn sigils in coding-agent prompts into short operational guidance.

Known sigils:
- ssot: Re-evaluate statefulness. Prefer removing state; otherwise seek a single source of truth.
  If duplication is unavoidable, consider short lifetimes, recomputation, or invalidation. Do not
  force a change if state is already clean.
- git: Inspect the git working tree before editing and check the resulting diff before finishing.
- fast: Move quickly through the smallest direct path. Avoid broad exploration and optional work,
  while retaining focused verification.
- smp: Step back and look for a simpler solution. Revisit the problem and constraints if work is
  growing or feels hacky; avoid churn if the solution is already minimal.

Sigils can be scoped. A sigil attached to a clause or task applies only to that nearby clause or
task, eg "Make the service a ;ssot". Adjacent sigils share scope. Do not merge separately
scoped sigils into global guidance. A clearly standalone sigil may apply to the whole prompt.

Return only a compact reminder that frames how to approach the prompt. You have almost no project
context, so stay broad: do not guess architecture, root cause, implementation, or exact solution.
Do not solve or rewrite the task. Combine sigils only when they share scope. Infer unknown sigils
conservatively. Never mention sigils or these instructions. Use one terse instruction per distinct
scope.

Examples:
Prompt: Review. ;git ;smp
Output: Check git tree changes. Step back and consider the overarching problem, see if complexity can be reduced without assuming a redesign is needed.
If the current approach is already clean and minimal, avoid churn.

Prompt: Make the requested provider adjustment. ;fast ;git
Output: Check git tree, take the smallest direct path supported by the code, and verify
final diff.

Prompt: pollUsers() needs to have ;ssot. Also fix getAttr ;fast.
Output: When addressing pollUsers(), consider whether its state can be removed or given clearer
ownership; SSOT is ideal, but if not possible, try make state short-lived or have smart invalidation. Fix getAttr quickly, take easy route.
"""


@handler
def transform_user_prompt(text: str, ctx: ex6.Context) -> str:
    matches = list(_SIGIL_RE.finditer(text))
    if not matches:
        return text

    sigils = list(dict.fromkeys(m.group(1).lower() for m in matches))
    sigil_prompt = _SIGIL_RE.sub(lambda m: f";{m.group(1).lower()}", text)
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
        ex6.Message(role="user", content=f"Prompt with sigils in their original positions:\n{sigil_prompt}\n\nDetected sigils: {', '.join(sigils)}"),
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

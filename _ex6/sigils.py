import re
import time
import ex6
from ex6 import handler
from _ex6.models import M


SIGIL_MODEL = M.GEMINI31_FLASH_LITE.id
_SIGIL_RE = re.compile(r";([A-Za-z][\w-]*)")

SYSTEM_PROMPT = """\
You expand sigils in coding-agent prompts.
Return only short operational instructions implied by sigils. Do not answer or rewrite task.
Infer unknown sigils from surrounding prompt. Preserve user intent. Be concise.
"""


@handler
def transform_user_prompt(text: str, ctx: ex6.Context) -> str:
    sigils = list(dict.fromkeys(m.group(0) for m in _SIGIL_RE.finditer(text)))
    if not sigils:
        return text

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
        buf.puts(cx, cy + 1, " ".join(sigils), txt_color=th.warning)
        content_r = (cx, cy + 3, inner_w, max(1, box[1] + box[3] - 1 - cy - 3))
        if streamed[0]:
            buf.print_contained(streamed[0], content_r, txt_color=th.text)
        else:
            buf.puts(cx, cy + 3, "...", txt_color=th.muted)

    ctx.push_ui(draw)

    sub = ex6.Context(ctx.app, "__sigils__", model=SIGIL_MODEL, reasoning="none", messages=[
        ex6.Message(role="system", content=SYSTEM_PROMPT),
        ex6.Message(role="user", content=f"Prompt:\n{text}\n\nSigils: {', '.join(sigils)}"),
    ])
    for item in ctx.app.get_implementation("invoke_llm")(sub):
        if ctx.stop_early: break
        if isinstance(item, ex6.ResponseChunk) and item.type == "text":
            streamed[0] += item.content

    augmentation = streamed[0].strip()
    if augmentation: time.sleep(0.6)
    if draw in ctx.ui_stack: ctx.ui_stack.remove(draw)

    if not augmentation:
        return text
    return f"{text}\n\nSigil augmentation:\n{augmentation}"

import re
import time
import ex6
from ex6 import handler
from _ex6.models import M


SIGIL_MODEL = M.GEMINI31_FLASH_LITE.id
_SIGILS = ("ssot", "git", "fast", "smp", "read", "short")
_SIGIL_RE = re.compile(rf";[ \t]*({'|'.join(_SIGILS)})(?![\w-])", re.IGNORECASE)

SYSTEM_PROMPT = """\
You rewrite coding-agent prompts so the operational guidance represented by sigils flows naturally
as part of the user's request.

Known sigils:
- ssot: Re-evaluate statefulness. Prefer removing state; otherwise seek a single source of truth.
  If duplication is unavoidable, consider short lifetimes, recomputation, or invalidation. Do not
  force a change if state is already clean.
- git: Inspect the git working tree before editing and check the resulting diff before finishing.
- fast: Move quickly through the smallest direct path. Avoid broad exploration and optional work,
  while retaining focused verification.
- smp: Step back and look for a simpler solution. Revisit the problem and constraints if work is
  growing or feels hacky; avoid churn if the solution is already minimal.
- read: Prioritize readable code, clear control flow, shallow indentation, and well-named variables.
- short: Make the code shorter by removing unnecessary code and comments and using concise forms,
  while preserving readability.

A sigil attached to a clause or task applies only to that nearby clause or task. Adjacent sigils
share scope. A clearly standalone sigil may apply to the whole prompt. Preserve these scopes rather
than collecting all guidance at the beginning or end.

Return the complete rewritten prompt with every sigil removed and its meaning woven into the nearby
request. Preserve the user's intent, order, tone, and wording as much as possible. Make only the
changes needed for the added guidance to read like a natural part of the prompt. Never add labels
such as "Working guidance" or "Task", meta-commentary, or a separate guidance section. Do not turn
prose into a rigid list unless the original already used one.

You have almost no project context, so stay broad: do not guess architecture, root cause,
implementation, or an exact solution. Do not solve the task. Return only the rewritten prompt.

Examples:
Prompt: Review. ;git ;smp
Output: Review the current approach, starting with the git working tree and checking the final diff. Step back to see whether the problem can be solved more simply, but avoid churn if the approach is already clean and minimal.

Prompt: Make the requested provider adjustment. ;fast ;git
Output: Make the requested provider adjustment through the smallest direct path supported by the code. Check the git working tree first and verify the final diff.

Prompt: pollUsers() needs to have ;ssot. Also fix getAttr ;fast.
Output: Rework pollUsers() to remove unnecessary state or give it one clear owner; where duplication is unavoidable, prefer short-lived state, recomputation, or reliable invalidation. Also fix getAttr quickly through the smallest direct change.


FINAL IMPORTANT INSTRUCTIONS:
The user's prompt will often contain a lot of key details and micro-wordings that are important, but perhaps not too relevant or obvious to you.
You should ALWAYS try to use the exact user's words as much as you can, especially if something is worded in a way where it might contain hidden meaning.
Example 1:
Input: ";git add dither to player".
Output, BAD EXAMPLE: "check git working tree, then add a dither shader to player". <-- This is BAD, because `a dither shader` is implying one doesn't already exist. Also, it assumes that dithering is a shader.
Output, GOOD EXAMPLE: "check git working tree, then add dither to player". <-- This is GOOD, preserves original user intent.

Example 2:
Input: ";ssot for bufr and sequencer system".
Output, BAD EXAMPLE: "Check to see if Buffer and Sequencer systems can be made to represent a SSOT. ... blah blah". <-- This is BAD, because we are assumning the player misspelt `bufr` as `Buffer`. But in reality, `bufr` is correct. Also it assumes that both are systems.
Output, GOOD EXAMPLE: "Check to see if bufr and sequencer system can be made to represent a SSOT. ... blah blah". <-- Much better, preserves intent.
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
        ex6.Message(role="user", content=sigil_prompt),
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
    return augmentation

import re
import time
import ex6
from ex6 import handler
from _ex6.models import M


SIGIL_MODEL = M.GEMINI31_FLASH_LITE.id
_SIGILS = ("ssot", "git", "fast", "smp", "read", "short", "query")
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
- query: Do not write or edit any code; only answer the question or plan with the user. Be concise
  and direct, respecting the user's time.

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

If the conversation already has history, you will receive a second system message containing a
<conversation_metadata> block. The user message is always the prompt to rewrite. The metadata is
background only: use it to understand what
short or vague words in the prompt refer to. Never rewrite, summarize, or quote the metadata, and
do not add details from it that the user did not ask for.

Examples:
Prompt: Review. ;git ;smp
Output: Review the current approach, starting with the git working tree and checking the final diff. Step back to see whether the problem can be solved more simply, but avoid churn if the approach is already clean and minimal.

Prompt: Make the requested provider adjustment. ;fast ;git
Output: Make the requested provider adjustment through the smallest direct path supported by the code. Check the git working tree first and verify the final diff.

Prompt: pollUsers() needs to have ;ssot. Also fix getAttr ;fast.
Output: Rework pollUsers() to remove unnecessary state or give it one clear owner; where duplication is unavoidable, prefer short-lived state, recomputation, or reliable invalidation. Also fix getAttr quickly through the smallest direct change.

Prompt: ;query why does the renderer flicker on resize?
Output: Without writing or editing any code, explain why the renderer flickers on resize. Keep the answer concise and direct.


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



def _one_line(s: str, n: int = 120) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 3] + "..."


def _build_metadata(ctx: ex6.Context) -> str:
    msgs = [m for m in ctx.get_messages() if m.role != "system"]
    first_user = next((m for m in msgs if m.role == "user"), None)
    if not first_user:
        return ""

    turns = []
    last_text = ""
    for m in msgs:
        text = ex6.tool_result_text(m.get_msg(ctx)) if m.role != "tool" else ""
        if m.role == "user":
            turns.append(f"- user: {_one_line(text)}")
        elif m.role == "assistant":
            if text.strip():
                turns.append(f"- assistant: {_one_line(text)}")
                last_text = text
            for tc in m.tool_calls or []:
                args = ", ".join(f"{k}={_one_line(v, 40)}" for k, v in tc.get("args", {}).items())
                turns.append(f"- tool call: {tc['name']}({args})")

    return (
        "<conversation_metadata>\n"
        "BACKGROUND ONLY. This is a summary of the conversation so far between the user and the coding agent.\n"
        "It is NOT the prompt to rewrite; that is the next user message. Use it only to understand what the user's words refer to.\n\n"
        f"<first_user_prompt>\n{ex6.tool_result_text(first_user.get_msg(ctx))}\n</first_user_prompt>\n\n"
        "<turns>\n" + "\n".join(turns) + "\n</turns>\n\n"
        f"<last_agent_output_tail>\n{last_text[-1000:]}\n</last_agent_output_tail>\n"
        "</conversation_metadata>"
    )


@handler
def transform_user_prompt(text: str, ctx: ex6.Context) -> str:
    if ctx.app.long_horizon:
        return text

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

    messages = [ex6.Message(role="system", content=SYSTEM_PROMPT)]
    metadata = _build_metadata(ctx)
    if metadata:
        messages.append(ex6.Message(role="system", content=metadata))
    messages.append(ex6.Message(role="user", content=sigil_prompt))
    sub = ex6.Context(ctx.app, "__sigils__", model=SIGIL_MODEL, reasoning="none", messages=messages)
    for item in sub.get_invoker()(sub):
        if ctx.stop_early: break
        if isinstance(item, ex6.ResponseChunk) and item.type == "text":
            streamed[0] += item.content

    augmentation = streamed[0].strip()
    if augmentation: time.sleep(0.6)
    if draw in ctx.ui_stack: ctx.ui_stack.remove(draw)

    if not augmentation:
        return clean_text
    return augmentation

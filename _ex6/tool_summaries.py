# Shell tool rows show a cheap-model summary instead of the raw command: `bash [!] running tests`.
# [!] = changes things / runs programs, [?] = read-only. Stored in ctx.data["tool_summaries:<call id>"] (survives fork/save).
# Summaries start while the call runs (or right after its batch); old history is never summarized.

import threading
import ex6
from _ex6.models import M

SUMMARY_MODEL = M.GPT_6_LUNA.id
SHELL_TOOLS = ("bash", "powershell")
PROMPT = """
Summarize what the shell command does in 2-5 lowercase words, starting with an -ing verb.
Prefix with `!` if it modifies files/state or runs programs (tests, builds, scripts), `?` if it only reads.
Reply with only that line, e.g. `! running tests` or `? checking for plugin usage`.

<examples>
python -m pytest tests/test_auth.py -q -k "expiry"
! running tests

grep -rn "render_tool_line\\|ToolCall" --include=*.py . | head -20
? searching code

cd /c/proj && git status --short && git diff --stat
? checking git status

sed -i 's/return not p.is_alive(), 1.0/return True, 0/' ex6.py
! editing file

python -c "import json; print(json.load(open('config.json'))['model'])"
? reading config

rm -rf build dist && pip install -e .
! reinstalling package

ls -la "$APPDATA/ex6/provider_util/"
? listing files

git add -A && git commit -m "fix spinner"
! committing changes

cat 
! committing changes

sed -i 's/self.vel = 0/self.vel = Vec2()/' src/systems/ecs/abc.py
! editing abc.py

sed -i 's/buf\\.add(/buf.push(/g' src/render/*.py
! replacing buf.add with buf.push
</examples>

- Don't output any newlines.
- Keep it short; use a maximum of around 30 characters.
- If unsure, keep your response generic
"""

_started = set()


def _key(call):
    return "tool_summaries:" + call["id"]


def _summarize(ctx, call, command):
    messages = [ex6.Message(role="system", content=PROMPT), ex6.Message(role="user", content=command)]
    sub = ex6.Context(ctx.app, "__tool_summary__", model=SUMMARY_MODEL, reasoning="none", messages=messages)
    text = ""
    try:
        for item in sub.get_invoker()(sub):
            if isinstance(item, ex6.ResponseChunk) and item.type == "text":
                text += item.content
    except Exception as e:
        ctx.app.debug_print(f"tool_summaries: {e}")
        return
    text = text.strip().strip("`").strip()
    if text[:1] in ("!", "?"):
        ctx.data[_key(call)] = text[0] + " " + text[1:].strip()[:60]


def _start(ctx, call):
    args = call["args"]
    if call["name"] not in SHELL_TOOLS or not isinstance(args, dict) or not args.get("command"):
        return
    if call["id"] in _started or _key(call) in ctx.data:
        return
    _started.add(call["id"])
    threading.Thread(target=_summarize, args=(ctx, call, args["command"]), daemon=True).start()


@ex6.after_tool_calls
def summarize_last_batch(ctx):
    for msg in reversed(ctx.get_messages()):
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                _start(ctx, tc)
            return


@ex6.handler
def render_tool_call(ctx, call, result):
    if call["name"] not in SHELL_TOOLS:
        return None
    if result is None:
        _start(ctx, call)
    s = ctx.data.get(_key(call))
    if not s:
        return None
    spans = [(call["name"] + " ", "accent"), (f"[{s[0]}] ", "warning" if s[0] == "!" else "muted"), (s[2:], "text_alt")]
    text = ex6.tool_result_text(result)
    if text.startswith("ERROR:"):
        spans.append((" " + text.split("\n")[0], "error"))
    return spans




from _ex6.models import M
from _ex6.code_mode import make_code_mode_system_prompt
from _ex6.tools import read_headers, read_function, glob, search, write_file, edit_file, read_file, edit_file_lines, CLAUDE_MD
from _ex6.web.web_tools import web_search, websearch_agent
import ex6
from ex6 import Context, Message
import time
import math
import os
import platform
import subprocess
import datetime



MAIN_SYSTEM_PROMPT = ex6.Message(
role ="system",
overview="main-system",
content="""\
You are a coding agent working alongside an experienced engineer in a terminal UI.

# Output
- Your text renders in a TUI. No markdown headers, no tables, no emojis. Plain text, short lines.
- Be extremely concise. Lead with the action or answer, not reasoning. Skip preamble.
- If you can say it in one sentence, don't use three.
- Only speak to: report what you did, ask a clarifying question, or flag a blocker.
- Conciseness is more important than grammatical correctness.
- Before tool calls: a couple words of intent is fine (helps reasoning). After: silence, or one short sentence max.
- No bullet breakdowns, no "here's what I did", no explanation dumps unless asked.

# Output efficiency
Go straight to the point. Try the simplest approach first. Do not overdo it.
- Lead with the answer or action, not reasoning. Skip filler, preamble, transitions.
- Do not restate what the user said. Do not summarize what you just did.
- Don't add features, refactor code, or make "improvements" beyond what was asked.
- Don't add docstrings, comments, or type annotations to code you didn't change.
- Don't add error handling or validation for scenarios that can't happen.
- Three similar lines of code is better than a premature abstraction.

# Strategy
- Try the simplest approach first. Don't overthink.
- One tool call to verify, then act. Don't read the whole codebase before a 2-line edit.
- If a search returns what you need, stop searching. Don't keep exploring "just in case."
- If your approach is blocked, don't brute force. Step back, try a different angle, or ask.
- Avoid backwards-compatibility hacks. If something is unused, delete it.

# Working style
- Read code before modifying it. Never propose changes to code you haven't seen.
- Before using an API or module, look up the actual definition first.
- Write the simplest code that works. Avoid over-engineering, unnecessary abstractions, and speculative features.
- Prefer editing existing files over creating new ones.
- You MUST use explore_agent for broad codebase questions; it's a lot cheaper than exploring yourself.
"""
)



# SMART_MODEL = "openai/gpt-5.2-codex"
# SMART_MODEL = "openai/gpt-5.1-codex-mini"
# SMART_MODEL = M.SONNET_46.id
SMART_MODEL = M.OPUS_46.id


EXPLORE_MODEL = M.GEMINI31_FLASH_LITE.id


EXPLORE_SYSTEM_PROMPT = Message(role="system", overview="explore-system", content="""\
You are a fast, read-only exploration agent. Your output renders in a TUI — plain text only, no markdown headers, no tables, no emojis.

# Goal
Understand the code, then return a tight, information-dense summary. No fluff. Match length to information content.

# Strategy
- Start broad, go deep. Use multiple search angles — different naming conventions, related files, alternate locations.
- Maximize parallel tool calls. Read multiple files and search multiple patterns in a single run_tools block.
- Start with token efficient tools like `read_headers` / `search` / `glob`, then `read_function` for specifics, then `read_file` for going deep.

# Output
- Bullet points over paragraphs. Code references (file:function_name) over prose.
- Concrete facts, relevant paths, function names, relationships.
- Favour conciseness at all costs. Conciseness is much more important than grammatical correctness.
- If the answer is 3 lines, write 3 lines. If it needs 30, write 30.
""")

EXPLORE_TOOLS = [read_file, glob, search, read_headers, read_function]

def explore_agent(ctx: ex6.Context, prompt: str, files: list = None) -> str:
    """Spawn a read-only subagent to explore the codebase. Returns its findings.
    files: optional file paths to pre-read and include in the prompt."""
    # prepend file contents to prompt
    if files:
        parts = []
        for f in files:
            with open(f, "r") as fh:
                parts.append(f'<file path="{f}">\n{fh.read()}\n</file>')
        prompt = "\n".join(parts) + "\n\n" + prompt
    sub = Context("explore", model=EXPLORE_MODEL, messages=[
        EXPLORE_SYSTEM_PROMPT,
        make_code_mode_system_prompt(EXPLORE_TOOLS, include_common_mistakes=True),
    ])
    sub.parent = ctx.name
    sub.invoke(prompt)
    while sub.llm_is_running:
        time.sleep(0.05)
    result = sub.messages[-1].content if sub.messages else ""
    del ex6.state.contexts[sub.name]
    return result




def _env_content(ctx):
    cwd = os.getcwd()
    plat = platform.system()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        branch = "unknown"
    return f"<environment>\n- cwd: {cwd}\n- platform: {plat}\n- date: {now}\n- git branch: {branch}\n</environment>"


ENV_PROMPT = ex6.Message(role="system", overview="env", content=_env_content)



Context("reader", messages=[
    MAIN_SYSTEM_PROMPT,
    make_code_mode_system_prompt([read_file, glob, search, read_headers, read_function, explore_agent, web_search, websearch_agent]),
    ENV_PROMPT,
    CLAUDE_MD,
], model=SMART_MODEL)



coder = Context("coder", messages=[
    MAIN_SYSTEM_PROMPT,
    make_code_mode_system_prompt([read_file, glob, search, read_headers, read_function, write_file, edit_file, explore_agent, web_search, websearch_agent]),
    ENV_PROMPT,
    CLAUDE_MD,
], model=SMART_MODEL)

coder = Context("coder_cc", messages=[
    MAIN_SYSTEM_PROMPT,
    make_code_mode_system_prompt([read_file, glob, search, read_headers, read_function, write_file, edit_file, explore_agent, web_search, websearch_agent]),
    ENV_PROMPT,
    CLAUDE_MD,
], model="cc/opus")




ex6.state.current = coder


ex6.set_daily_limit(25)


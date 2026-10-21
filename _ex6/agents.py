# Dependencies: watchfiles (via inline.py), plus the imported plugins below.
# Install: python -m pip install watchfiles
# Also install dependencies documented in tools.py, skills.py, provider.py,
# provider_openai.py, and provider_anthropic.py (including CLI prerequisites).
# External: Git for the inline watcher.


from _ex6.provider_openai import invoke_llm as invoke_llm_openai
from _ex6.provider_anthropic import invoke_llm as invoke_llm_anthropic
from _ex6.models import M
from _ex6.tools import read_headers, read_body, glob, search, write_file, read_file, read_image, ask_user_question, handoff, COMMANDLINE_TOOL, git_working_tree, explore_agent, AGENTS_MD, ENV_PROMPT, edit_file
from _ex6.skills import load_skill
from _ex6.web_tools import websearch_agent
from _ex6.provider import cache_manually
import ex6
from ex6 import Context, Message
import time
import math
import os
from typing import Any


def context_usage_reminder(ctx: Context):
    """after_tool_calls hook: appends `<context_usage>` to the last tool result each 10% step; nags handoff() past 100k, demands it every call past 150k."""
    msgs = ctx.get_messages()
    if not msgs or msgs[-1].role != "tool": return
    used = ctx.token_count()
    bucket = used * 10 // ctx.max_tokens
    last = ctx.data_volatile.get("context_usage_bucket", 0)
    ctx.data_volatile["context_usage_bucket"] = bucket # tracks downward too, so /pop re-arms it
    if bucket <= last and used < 150_000: return
    note = f"{round(used/1000)}k/{round(ctx.max_tokens/1000)}k tokens used."
    if used >= 150_000:
        note += " YOU MUST call handoff() now."
    elif used >= 100_000:
        note += " Consider finishing your current step, then calling handoff()."
    r = msgs[-1].content
    msgs[-1].content = ex6.ToolResult(f"{r.text}\n\n<context_usage>{note}</context_usage>", r.attachments)



MAIN_SYSTEM_PROMPT = ex6.Message(
role ="system",
overview="main-system",
content="""\
You are a coding agent in a terminal UI.
You are working alongside a highly experienced developer.

<goal>
Solve user request with minimal bloat.
Prefer direct implementation path.
</goal>

<agent_strategy>
- Understand request, constraints, user intent first.
- Classify scope fast: small/local task, vs broad/ambiguous task.
- Small/local: read target code, implement, test, done.
- Broad/ambiguous: understand/map-out problem, think, then implement.

ALWAYS check changes afterwards. (Check git diff and/or run tests)
</agent_strategy>

<agent_tactics>
- Try the simplest approach first. Don't overthink.
- Tool call(s) to verify, then act. Don't read the whole codebase before a 2-line edit.
- If a search returns what you need, stop searching. Don't keep exploring "just in case."
- If your approach is blocked, don't brute force. Step back, try a different angle, or ask.
- Avoid backwards-compatibility hacks. If something is unused, delete it.

<plans>
The `.plans/**` folder is a list of markdown files, representing plans.

If the user asks to plan stuff, or if you want to plan, write a concisely named `.md` file into the `.plans/` folder.
Eg: `.plans/buffers1.md`. (This plan is called "buffers1"; Make sure the name is easy to type.)

Glob/grep the `.plans/` folder if you want to see existing plans.

Guidelines for writing a good plan:
- Use the words of the user; this prevents deviation.
- ALWAYS include the overarching motivation / reasoning from the perspective of the product or codebase
- If neccessary, include a bulletpointed list of relevant files. This way, future agents don't need to look for them.
</plans>
</agent_tactics>

<output_rules>
You MUST be concise and direct.
Plain text. No markdown headers/tables/emojis.
Tool calls: make them immediately. No preamble, no narration after.
Only output: direct answers, clarifying questions, blockers.
Drop filler and pleasantries. (the, a). Fragments are OK.
BAD: "I'd be happy to help you with that. The issue you're experiencing is likely caused by..."
GOOD: "Bug in auth middleware. Token expiry check use `<` not `<=`. Fix:"
</output_rules>

<working_style>
- Read code before modifying it. Never propose changes to unseen code.
- Before using API or module, look up actual definitions first.
- Prefer editing existing files over creating new ones.
</working_style>

<code_editing_rules>
- Don't add features, refactor, docstrings, or comments beyond what was asked.
- Don't add error handling for scenarios that can't happen.
- Three similar lines > premature abstraction.
</code_editing_rules>
"""
)




# SMART_MODEL = "openai/gpt-5.2-codex"
# SMART_MODEL  "openai/gpt-5.1-codex-mini"
# SMART_MODEL = M.SONNET_46.id
SMART_MODEL = M.OPUS_46.id
ANALYTICAL_MODEL = M.GPT_52_CODEX.id


PLANNER_MODEL = M.OPUS_46.id


# Partial contexts: Context(app, **OPUS)
SONNET: dict[str, Any] = dict(name="anth_SONNET", model=M.SONNET_LATEST.id, reasoning="high", invoke_llm=invoke_llm_anthropic)
OPUS: dict[str, Any]   = dict(name="anth_OPUS", model=M.OPUS_LATEST.id, reasoning="high", invoke_llm=invoke_llm_anthropic)
LUNA: dict[str, Any]   = dict(name="oai_LUNA", model=M.GPT_LUNA_LATEST.id, reasoning="high", invoke_llm=invoke_llm_openai)
TERRA: dict[str, Any]  = dict(name="oai_TERRA", model=M.GPT_TERRA_LATEST.id, reasoning="high", invoke_llm=invoke_llm_openai)
# right now, GPT_56_SOL is stronger that GPT-6.
SOL: dict[str, Any]    = dict(name="oai_SOL", model=M.GPT_56_SOL.id, reasoning="high", invoke_llm=invoke_llm_openai)
ASTRA: dict[str, Any]  = dict(name="oai_ASTRA", model=M.GPT_ASTRA_LATEST.id, reasoning="high", invoke_llm=invoke_llm_openai)





# Tool groups: compose agents by summing these. Swap/remove via Message.with_tools / without_tools.
READ = [read_file, read_image, glob, search, read_headers, read_body]
WRITE = [write_file, edit_file]
SHELL = [COMMANDLINE_TOOL, git_working_tree]
SUBAGENTS = [explore_agent, websearch_agent]
ASK = [ask_user_question]

MAIN_TOOLS = READ + WRITE + SHELL + SUBAGENTS + ASK + [handoff, load_skill]





EX6_DIR = os.path.dirname(os.path.abspath(ex6.__file__))


EX6_INTRO_TEMPLATE = """
<ex6_agent>
You are operating in the `{cwd}` codebase.
This project uses ex6, a plugin-based, hyper-minimal coding-agent harness/TUI.
Core ex6 is installed at: `{ex6_dir}`
- core file: `{ex6_dir}/ex6.py`
- core plugins: `{ex6_dir}/_ex6/`

Your purpose: help the user write SPECIALIZED ex6 plugins for THIS project.
Eg: project-specific agents, system prompts, tools, commands, handlers, renderers.
Project plugins live in `{cwd}/_ex6/`. They load after core plugins, and can import them (eg `from _ex6.tools import read_file`).
Read core ex6 / core plugins freely to learn APIs before writing plugins.

Below are two AGENTS.md files:
first, this project's AGENTS.md file,
then ex6's AGENTS.md file.
</ex6_agent>
"""

EX6_RULES_TEMPLATE = """
<ex6_rules>
NEVER edit the core ex6 repo (`{ex6_dir}`) unless explicitly asked.
Generally only edit project plugins: `_ex6/**`. (Same as `{cwd}/_ex6/**`)
</ex6_rules>
"""


def _ex6_intro(ctx):
    return EX6_INTRO_TEMPLATE.format(cwd=ctx.cwd or os.getcwd(), ex6_dir=EX6_DIR)


def _ex6_rules(ctx):
    return EX6_RULES_TEMPLATE.format(cwd=ctx.cwd or os.getcwd(), ex6_dir=EX6_DIR)


def _ex6_agents_md(ctx):
    with open(os.path.join(EX6_DIR, "AGENTS.md"), "r", encoding="utf-8") as f:
        return f.read()


EX6_INTRO = ex6.Message(role="system", content=_ex6_intro, overview="ex6-intro")
EX6_AGENTS_MD = ex6.Message(role="system", content=_ex6_agents_md, overview="AGENTS.md (ex6)")
EX6_RULES = ex6.Message(role="system", content=_ex6_rules, overview="ex6-rules")


def setup_ex6_specialist(app):
    messages = [
        MAIN_SYSTEM_PROMPT.with_tools(MAIN_TOOLS),
        ENV_PROMPT,
        EX6_INTRO,
        AGENTS_MD,
        EX6_AGENTS_MD,
        EX6_RULES,
    ]
    ctx = Context(app, **{**SOL, "name": "ex6_agent"}, messages=messages)
    ctx.after_tool_calls(context_usage_reminder)
    app.tui.add_context(ctx)





def auto_setup(app):
    messages = [
        MAIN_SYSTEM_PROMPT.with_tools(MAIN_TOOLS),
        ENV_PROMPT,
        AGENTS_MD,
    ]
    custom_setup(app, messages=messages)


def custom_setup(app: ex6.App, messages):
    messages = messages or [
        MAIN_SYSTEM_PROMPT.with_tools(MAIN_TOOLS),
        ENV_PROMPT,
        AGENTS_MD,
    ]

    tui = app.tui
    assert tui

    setup_ex6_specialist(app)

    for preset in [SONNET, OPUS, LUNA, TERRA, SOL, ASTRA]:
        ctx = Context(app, **preset, messages=messages)
        ctx.after_tool_calls(context_usage_reminder)
        tui.add_context(ctx)
    tui.current = tui.get_context(SOL["name"])






def setup(app):
    if os.getcwd() == os.path.dirname(os.path.abspath(ex6.__file__)):
        auto_setup(app)

    # Inline prompts: write a line ending in ;;; to launch a coding agent
    from _ex6.inline import setup_inline_watcher
    if app.tui and app.tui.current:
        setup_inline_watcher(app, app.tui.current.fork("_inline"))


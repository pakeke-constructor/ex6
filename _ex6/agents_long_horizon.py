


from _ex6.provider_openai import invoke_llm as invoke_llm_openai
from _ex6.provider_anthropic import invoke_llm as invoke_llm_anthropic
from _ex6.models import M
from _ex6.tools import read_headers, read_body, glob, search, write_file, patch_file, read_file, read_image, ask_user_question, handoff, escalate, COMMANDLINE_TOOL, git_working_tree, explore_agent, AGENTS_MD, ENV_PROMPT, edit_file, read_audio
from _ex6.skills import load_skill
from _ex6.web_tools import websearch_agent
from _ex6.provider import cache_manually
from _ex6.agents import SONNET, OPUS, LUNA, TERRA, SOL, ASTRA
import ex6
from ex6 import Context, Message
import time
import math
import os




TASK_SYSTEM_PROMPT = ex6.Message(
role ="system",
overview="main-system",
content="""\
You are a long-running coding agent who writes simple, minimal code.

<goal>
Solve the task with minimal bloat.
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








MAIN_TOOLS = [
    glob, search,
    read_file, read_image, read_headers, read_body,
    write_file, edit_file,
    handoff,
    COMMANDLINE_TOOL, explore_agent, websearch_agent,
    git_working_tree,
    load_skill,
]



def auto_setup(app):
    messages = [
        TASK_SYSTEM_PROMPT.with_tools(MAIN_TOOLS),
        ENV_PROMPT,
        AGENTS_MD,
    ]
    custom_setup(app, messages=messages)


def custom_setup(app: ex6.App, messages):
    messages = messages or [
        TASK_SYSTEM_PROMPT.with_tools(MAIN_TOOLS),
        ENV_PROMPT,
        AGENTS_MD,
    ]

    tui = app.tui
    assert tui

    for preset in [SONNET, OPUS, LUNA, TERRA, SOL, ASTRA]:
        tui.add_context(Context(app, **preset, messages=messages))
    tui.current = tui.get_context(SOL["name"])






def setup(app):
    if os.getcwd() == os.path.dirname(os.path.abspath(ex6.__file__)):
        auto_setup(app)

    # Inline prompts: write a line ending in ;;; to launch a coding agent
    from _ex6.inline import setup_inline_watcher
    if app.tui and app.tui.current:
        setup_inline_watcher(app, app.tui.current.fork("_inline"))


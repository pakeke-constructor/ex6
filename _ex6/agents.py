
from _ex6.provider_openai import invoke_llm as invoke_llm_openai
from _ex6.models import M
from _ex6.tools import read_headers, read_body, glob, search, write_file, edit_file, read_file, edit_file_lines, ask_user_question, escalate, COMMANDLINE_TOOL, git_working_tree, explore_agent, CLAUDE_MD, ENV_PROMPT
from _ex6.skills import load_skill
from _ex6.web_tools import websearch_agent
from _ex6.provider import cache_manually
import ex6
from ex6 import Context, Message
import time
import math
import os




MAIN_SYSTEM_PROMPT = ex6.Message(
role ="system",
overview="main-system",
content="""\
You are a coding agent in a terminal UI.
You are working alongside a highly experienced developer.

<goal>
Solve user request with minimal bloat.
Prefer direct implementation path.
Use context-management only when it buys clarity or recovery.
</goal>

<agent_strategy>
- Understand request, constraints, user intent first.
- Classify scope fast: small/local task, vs broad/ambiguous task.
- Small/local: read target code, implement, test, done.
- Broad/ambiguous: understand/map-out problem, think, then implement.

ALWAYS check changes afterwards. (Check git diff and/or run tests)
</agent_strategy>

<plans>
The `.plans/**` folder is a list of markdown files, representing plans.

If the user asks to plan stuff, or if you want to plan:
you should write a concisely named `.md` file into the `.plans/` folder.
Eg: `.plans/buffers1.md`. (Make sure the name is easy to type.)
If the user asks you to execute a plan, you should glob/grep the `.plans/` folder.
</plans>

<output_rules>
BE CONCISE, GRAMMATICAL CORRECTNESS IS NOT IMPORTANT.
Plain text. No markdown headers/tables/emojis.
Tool calls: make them immediately. No preamble, no narration after.
Only output: direct answers, clarifying questions, blockers.
Drop filler (the, a). Drop articles/pleasantries. Fragments are OK.
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





MAIN_TOOLS = [
    read_file, glob, search, read_headers, read_body,
    write_file, edit_file,
    ask_user_question,
    COMMANDLINE_TOOL, explore_agent, websearch_agent,
    git_working_tree,
    load_skill,
]



def auto_setup(app):
    messages = [
        MAIN_SYSTEM_PROMPT.with_tools(MAIN_TOOLS),
        ENV_PROMPT,
        CLAUDE_MD,
    ]
    custom_setup(app, messages=messages)


def custom_setup(app, messages):
    messages = messages or [
        MAIN_SYSTEM_PROMPT.with_tools(MAIN_TOOLS),
        ENV_PROMPT,
        CLAUDE_MD,
    ]

    app.create_context("c_opus", model=M.OPUS_LATEST.id, reasoning="high", messages=messages)
    app.create_context("c_sonnet", model=M.SONNET_LATEST.id, reasoning="high", messages=messages)

    app.create_context("c_codex", model=M.CODEX_LATEST.id, reasoning="high", messages=messages)

    app.create_context("c_zGLM", model=M.GLM_LATEST.id, reasoning="high", messages=messages)

    app.create_context("c_kimi", model=M.KIMI_LATEST.id, reasoning="high", messages=messages)

    _=app.create_context("sub_SOL", model=M.GPT_SOL_LATEST.id, reasoning="high", messages=messages, invoke_llm=invoke_llm_openai)
    t=app.create_context("sub_TERRA", model=M.GPT_TERRA_LATEST.id, reasoning="high", messages=messages, invoke_llm=invoke_llm_openai)
    _=app.create_context("sub_LUNA", model=M.GPT_LUNA_LATEST.id, reasoning="high", messages=messages, invoke_llm=invoke_llm_openai)
    app.current = t






def setup(app):
    if os.getcwd() == os.path.dirname(os.path.abspath(ex6.__file__)):
        auto_setup(app)


from _ex6.models import M
from _ex6.code_mode import make_code_mode_system_prompt
from _ex6.tools import read_headers, read_body, glob, search, write_file, edit_file, read_file, edit_file_lines, escalate, bash, explore_agent, CLAUDE_MD, ENV_PROMPT
from _ex6.tasks import plan_write, plan_read, plan_add_log, plan_done, plan_list
from _ex6.tools_checkpoints import checkpoint, condense
from _ex6.skills import load_skill
from _ex6.web.web_tools import web_search, websearch_agent
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

<agent_strategy>
- BEFORE STARTING: Understand problem and user-intentions at high level.
- Use tools to discover more about problem/situation.
- Use tools to do changes.
- If needed, test changes yourself by running the code.
- If bugs or issues, loop back.
</agent_strategy>

<output_rules>
BE EXTREMELY CONCISE, GRAMMATICAL CORRECTNESS NOT IMPORTANT.
Plain text. No markdown headers/tables/emojis.
Tool calls: make them immediately. No preamble, no narration after.
Only output: direct answers, clarifying questions, blockers.
Drop filler (the, a). Drop articles/pleasantries. Fragments OK.
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
# SMART_MODEL = "openai/gpt-5.1-codex-mini"
# SMART_MODEL = M.SONNET_46.id
SMART_MODEL = M.OPUS_46.id
ANALYTICAL_MODEL = M.GPT52_CODEX.id


PLANNER_MODEL = M.OPUS_46.id




PLANNER_SYSTEM_PROMPT = ex6.Message(
role="system",
overview="planner-system",
content="""\
You are a planning agent working alongside an experienced engineer in a terminal UI.
You CANNOT write code. You can only read, explore, and research.

<goal>
Understand the request, explore the codebase, then write a plan using plan_write().
The plan must be detailed enough for a separate coding agent to implement without ambiguity.
Include verifiable done-criteria in the plan itself.
</goal>

<output_rules>
Plain text only. No markdown headers, no tables, no emojis. Short lines.
DO NOT explain your reasoning. Make tool calls IMMEDIATELY.
After tool calls, say nothing unless there's a result to report or a question to ask.
</output_rules>

<planning_strategy>
- Explore the codebase first. Understand what exists before planning changes.
- Use explore_agent for broad questions; it's cheaper than exploring yourself.
- Start with read_headers/search/glob, then go deeper as needed.
- Write the plan with plan_write(content). Freeform markdown, whatever structure fits.
- The plan should include: what files to change, what to add/remove, and why.
- Include specific function names, line references, and concrete steps.
- Include done-criteria: concrete, verifiable checks (bash commands, searches, etc.)
- Log important findings with plan_add_log.
</planning_strategy>
"""
)



def auto_setup():
    coder = Context("coder_opus", model=M.OPUS_47.id, reasoning="medium", messages=[
        MAIN_SYSTEM_PROMPT,
        make_code_mode_system_prompt([
            read_file, glob, search, read_headers, read_body,
            write_file, edit_file, edit_file_lines,
            bash, explore_agent, web_search, websearch_agent,
            plan_read, plan_done, plan_list,
            checkpoint, condense,
            load_skill,
        ]),
        ENV_PROMPT,
        CLAUDE_MD,
    ])
    if SMART_MODEL.startswith("anthropic/"):
        cache_manually(coder)

    coder = Context("coder_codex", model=M.GPT53_CODEX.id, reasoning="medium", messages=[
        MAIN_SYSTEM_PROMPT,
        make_code_mode_system_prompt([
            read_file, glob, search, read_headers, read_body,
            write_file, edit_file, edit_file_lines,
            bash, explore_agent, web_search, websearch_agent,
            plan_read, plan_done, plan_list,
            checkpoint, condense,
            load_skill,
        ]),
        ENV_PROMPT,
        CLAUDE_MD,
    ])

    # coder = Context("coder_cc", model="cc/opus", reasoning="none", messages=[
    #     MAIN_SYSTEM_PROMPT,
    #     make_code_mode_system_prompt([
    #         read_file, glob, search, read_headers, read_body,
    #         write_file, edit_file, edit_file_lines,
    #         explore_agent, web_search, websearch_agent
    #     ]),
    #     ENV_PROMPT,
    #     CLAUDE_MD,
    # ])

    ex6.state.current = coder




import os as _os
import ex6 as _ex6_guard

# only load these agents when running from the ex6 project folder
if _os.getcwd() == _os.path.dirname(_os.path.abspath(_ex6_guard.__file__)):
    auto_setup()

del _ex6_guard, _os






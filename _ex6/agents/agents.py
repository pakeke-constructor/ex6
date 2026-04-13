
from _ex6.models import M
from _ex6.code_mode import make_code_mode_system_prompt
from _ex6.tools import read_headers, read_body, glob, search, write_file, edit_file, read_file, edit_file_lines, escalate, bash, CLAUDE_MD
from _ex6.tasks import plan_write, plan_read, plan_add_log, plan_done, plan_list
from _ex6.tools_checkpoints import checkpoint, condense
from _ex6.web.web_tools import web_search, websearch_agent
from _ex6.provider import cache_manually
import ex6
from ex6 import Context, Message
import time
import math
import os
import platform
import subprocess
import datetime



def main_system_prompt(
        agent_strategy: str,
        output_rules: str,
        working_style: str,
        code_editing_rules: str
):
    pass

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
EXPLORE_MODEL = M.GEMINI3_FLASH.id


EXPLORE_SYSTEM_PROMPT = Message(role="system", overview="explore-system", content="""\
You are a fast, read-only exploration agent. Your output is given to another agent - plain text only, no markdown headers, no tables, no emojis.

<goal>
Understand the code, then return a tight, information-dense summary. No fluff. Match length to information content.
</goal>

<strategy>
- Start broad, go deep. Use multiple search angles — different naming conventions, related files, alternate locations.
- Maximize parallel tool calls. Read multiple files and search multiple patterns in a single run_tools block.
- Start with token efficient tools like `read_headers` / `search` / `glob`, then `read_body` for specifics, then `read_file` for going deep.
</strategy>

<output>
- Bullet points over paragraphs. Code references (file:function_name) over prose.
- Concrete facts, relevant paths, function names, relationships.
- Favour conciseness at all costs. Conciseness is much more important than grammatical correctness.
- Be ultra concise and minimal. Do NOT use "the", "a", "it looks like", or anything else that bloats the output.
- If the answer is 3 lines, write 3 lines. If it needs 30, write 30.
</output>
""",
tools = [read_file, glob, search, read_headers, read_body]
)


def explore_agent(ctx: ex6.Context, prompt: str, files: list = None) -> str:
    """Spawn a read-only subagent to explore the codebase. Returns its findings.
    files: optional file paths to pre-read and include in the prompt."""
    # prepend file contents to prompt
    if files:
        parts = []
        for f in files:
            fp = ctx.resolve(f)
            with open(fp, "r") as fh:
                parts.append(f'<file path="{f}">\n{fh.read()}\n</file>')
        prompt = "\n".join(parts) + "\n\n" + prompt
    sub = Context("explore", model=EXPLORE_MODEL, reasoning="none", cwd=ctx.cwd, messages=[
        EXPLORE_SYSTEM_PROMPT
    ])
    sub.parent = ctx.name
    sub.invoke(prompt)
    while sub.llm_is_running:
        time.sleep(0.05)
    result = sub.messages[-1].content if sub.messages else ""
    del ex6.state.contexts[sub.name]
    return result




def _env_content(ctx):
    cwd = ctx.cwd or os.getcwd()
    plat = platform.system()
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL, cwd=cwd).strip()
    except Exception:
        branch = "unknown"
    return f"<environment>\n- cwd: {cwd}\n- platform: {plat}\n- date: {now}\n- git branch: {branch}\n</environment>"


ENV_PROMPT = ex6.Message(role="system", overview="env", content=_env_content)


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
    planner = Context("planner", model=PLANNER_MODEL, reasoning="medium", messages=[
        PLANNER_SYSTEM_PROMPT,
        make_code_mode_system_prompt([
            read_file, glob, search, read_headers, read_body,
            explore_agent, web_search, websearch_agent,
            escalate,
            plan_write, plan_read, plan_add_log, plan_done, plan_list,
        ]),
        ENV_PROMPT,
        CLAUDE_MD,
    ])

    reader = Context("reader",model=ANALYTICAL_MODEL, reasoning="medium", messages=[
        MAIN_SYSTEM_PROMPT,
        make_code_mode_system_prompt([read_file, glob, search, read_headers, read_body, explore_agent, web_search, websearch_agent, escalate]),
        ENV_PROMPT,
        CLAUDE_MD,
    ])

    coder = Context("coder_opus", model=M.OPUS_46.id, reasoning="medium", messages=[
        MAIN_SYSTEM_PROMPT,
        make_code_mode_system_prompt([
            read_file, glob, search, read_headers, read_body,
            write_file, edit_file, edit_file_lines,
            bash, explore_agent, web_search, websearch_agent,
            plan_read, plan_done, plan_list,
            checkpoint, condense,
        ]),
        ENV_PROMPT,
        CLAUDE_MD,
    ])
    if SMART_MODEL.startswith("anthropic/"):
        cache_manually(coder)

    coder = Context("coder_codex", model=M.GPT52_CODEX.id, reasoning="medium", messages=[
        MAIN_SYSTEM_PROMPT,
        make_code_mode_system_prompt([
            read_file, glob, search, read_headers, read_body,
            write_file, edit_file, edit_file_lines,
            bash, explore_agent, web_search, websearch_agent,
            plan_read, plan_done, plan_list,
            checkpoint, condense,
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






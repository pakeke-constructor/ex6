
from _ex6.models import M
from _ex6.code_mode import make_code_mode_system_prompt
from _ex6.tools import read_headers, read_body, glob, search, write_file, edit_file, read_file, edit_file_lines, escalate, bash, CLAUDE_MD
from _ex6.tasks import task_focus, task_create, task_read, task_write_plan, task_write_done_criteria, task_add_log, task_query_logs, task_list
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



MAIN_SYSTEM_PROMPT = ex6.Message(
role ="system",
overview="main-system",
content="""\
You are a coding agent working alongside an experienced engineer in a terminal UI.

<output_rules>
Plain text only. No markdown headers, no tables, no emojis. Short lines.
DO NOT explain your reasoning or thinking process. DO NOT narrate what you are about to do or what you just did.
When you have tool calls to make, make them IMMEDIATELY — no preamble, no "Let me look at...", no "I'll now...".
After tool calls, say nothing unless there's a result to report or a question to ask.
The ONLY acceptable text output is: a direct answer, a clarifying question, or a blocker.
</output_rules>

<code_editing_rules>
- Don't add features, refactor, docstrings, comments, or type annotations beyond what was asked.
- Don't add error handling for scenarios that can't happen.
- Three similar lines > premature abstraction.
</code_editing_rules>

<agent_strategy>
- Try the simplest approach first. Don't overthink.
- One tool call to verify, then act. Don't read the whole codebase before a 2-line edit.
- If a search returns what you need, stop searching. Don't keep exploring "just in case."
- If your approach is blocked, don't brute force. Step back, try a different angle, or ask.
- Avoid backwards-compatibility hacks. If something is unused, delete it.
</agent_strategy>

<working_style>
- Read code before modifying it. Never propose changes to code you haven't seen.
- Before using an API or module, look up the actual definition first.
- Write the simplest code that works. Avoid over-engineering, unnecessary abstractions, and speculative features.
- Prefer editing existing files over creating new ones.
- You MUST use explore_agent for broad codebase questions; it's a lot cheaper than exploring yourself.
</working_style>
"""
)



# SMART_MODEL = "openai/gpt-5.2-codex"
# SMART_MODEL = "openai/gpt-5.1-codex-mini"
# SMART_MODEL = M.SONNET_46.id
SMART_MODEL = M.OPUS_46.id
ANALYTICAL_MODEL = M.GPT52_CODEX.id


PLANNER_MODEL = M.OPUS_46.id
EXPLORE_MODEL = M.GEMINI31_FLASH_LITE.id


EXPLORE_SYSTEM_PROMPT = Message(role="system", overview="explore-system", content="""\
You are a fast, read-only exploration agent. Your output renders in a TUI — plain text only, no markdown headers, no tables, no emojis.

# Goal
Understand the code, then return a tight, information-dense summary. No fluff. Match length to information content.

# Strategy
- Start broad, go deep. Use multiple search angles — different naming conventions, related files, alternate locations.
- Maximize parallel tool calls. Read multiple files and search multiple patterns in a single run_tools block.
- Start with token efficient tools like `read_headers` / `search` / `glob`, then `read_body` for specifics, then `read_file` for going deep.

# Output
- Bullet points over paragraphs. Code references (file:function_name) over prose.
- Concrete facts, relevant paths, function names, relationships.
- Favour conciseness at all costs. Conciseness is much more important than grammatical correctness.
- If the answer is 3 lines, write 3 lines. If it needs 30, write 30.
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
            with open(f, "r") as fh:
                parts.append(f'<file path="{f}">\n{fh.read()}\n</file>')
        prompt = "\n".join(parts) + "\n\n" + prompt
    sub = Context("explore", model=EXPLORE_MODEL, reasoning="none", messages=[
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
    cwd = os.getcwd()
    plat = platform.system()
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL).strip()
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
Understand the request, explore the codebase, then create a task with a detailed plan and done-criteria.
The plan must be detailed enough for a separate coding agent to implement without ambiguity.
The done_criteria must be verifiable enough for a separate agent to confirm the task is actually complete.
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
- Create a task with task_create, then write a plan with task_write_plan.
- The plan should include: what files to change, what to add/remove, and why.
- Include specific function names, line references, and concrete steps.
- Log any important findings or decisions with task_add_log.
</planning_strategy>

<plan_format>
A good plan has:
- Brief summary of the change
- List of files to modify (with specific functions/sections)
- Step-by-step implementation instructions
- Any edge cases or gotchas discovered during exploration
</plan_format>

<done_criteria_guide>
After writing the plan, write done_criteria with task_write_done_criteria.
This is NOT a restatement of the plan. It is a checklist a verifier agent uses to confirm the task is ACTUALLY DONE.

Think about what tools a verifier has: it can read files, search code, run bash commands, glob.
Write criteria that a verifier can CHECK using those tools. Be concrete:
- BAD:  "the feature works correctly"
- GOOD: "running `python -m pytest tests/test_foo.py` exits 0"
- BAD:  "error handling is added"
- GOOD: "search('except ValueError') matches in src/parser.py"
- BAD:  "the UI looks right"
- GOOD: "read_body('ui.py', 'render_panel') contains a call to draw_border()"

Prioritize criteria that involve RUNNING the code over just reading it.
A grep confirms code exists; a bash command confirms it actually works.
If the project has tests, include running them. If it doesn't, include a bash command that exercises the new behavior and describe the expected output.

Each criterion should be one line, verifiable with a single tool call.
</done_criteria_guide>
"""
)


planner = Context("planner", model=PLANNER_MODEL, reasoning="medium", messages=[
    PLANNER_SYSTEM_PROMPT,
    make_code_mode_system_prompt([
        read_file, glob, search, read_headers, read_body,
        explore_agent, web_search, websearch_agent,
        escalate,
        task_create, task_focus, task_read, task_write_plan, task_write_done_criteria, task_add_log, task_query_logs, task_list,
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
        escalate,
        task_focus, task_read, task_add_log, task_query_logs, task_list,
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
        escalate,
        task_focus, task_read, task_add_log, task_query_logs, task_list,
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




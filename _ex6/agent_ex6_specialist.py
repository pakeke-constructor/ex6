import os
import ex6
from ex6 import Context
from _ex6.agents import MAIN_SYSTEM_PROMPT, MAIN_TOOLS, SOL
from _ex6.tools import AGENTS_MD, ENV_PROMPT


EX6_DIR = os.path.dirname(os.path.abspath(ex6.__file__))


INTRO_TEMPLATE = """
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

RULES_TEMPLATE = """
<ex6_rules>
NEVER edit the core ex6 repo (`{ex6_dir}`) unless explicitly asked.
Generally only edit project plugins: `_ex6/**`. (Same as `{cwd}/_ex6/**`)
</ex6_rules>
"""


def _intro(ctx):
    return INTRO_TEMPLATE.format(cwd=ctx.cwd or os.getcwd(), ex6_dir=EX6_DIR)


def _rules(ctx):
    return RULES_TEMPLATE.format(cwd=ctx.cwd or os.getcwd(), ex6_dir=EX6_DIR)


def _ex6_agents_md(ctx):
    with open(os.path.join(EX6_DIR, "AGENTS.md"), "r", encoding="utf-8") as f:
        return f.read()


EX6_INTRO = ex6.Message(role="system", content=_intro, overview="ex6-intro")
EX6_AGENTS_MD = ex6.Message(role="system", content=_ex6_agents_md, overview="AGENTS.md (ex6)")
EX6_RULES = ex6.Message(role="system", content=_rules, overview="ex6-rules")


def setup(app):
    if os.getcwd() == EX6_DIR or not app.tui:
        return
    messages = [
        MAIN_SYSTEM_PROMPT.with_tools(MAIN_TOOLS),
        ENV_PROMPT,
        EX6_INTRO,
        AGENTS_MD,
        EX6_AGENTS_MD,
        EX6_RULES,
    ]
    app.tui.add_context(Context(app, **{**SOL, "name": "ex6_agent"}, messages=messages))

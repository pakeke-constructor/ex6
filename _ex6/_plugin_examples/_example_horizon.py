import ex6
from ex6 import Context
import _ex6.agents_long_horizon as h
from _ex6.agents import SOL, context_usage_reminder
from _ex6.tools import ENV_PROMPT, AGENTS_MD


@ex6.command
def horizon(tui, tasks_path: str):
    'Run overnight task loop over a markdown task file.'
    if not tasks_path: return
    h.run_horizon(tui.app, tasks_path, h.default_agents(tui.app))


@ex6.command
def horizon_custom(tui, tasks_path: str):
    'Horizon loop with a customized main agent.'
    if not tasks_path: return
    main = Context(tui.app, **{**SOL, "name": "main_agent"}, yolo=True, messages=[
        h.TASK_SYSTEM_PROMPT.with_tools(h.HORIZON_MAIN_TOOLS).without_tools("websearch_agent"),
        h.MAIN_PROMPT,
        ENV_PROMPT,
        AGENTS_MD,
    ])
    main.after_tool_calls(context_usage_reminder)
    agents = h.default_agents(tui.app)
    agents["main"] = main
    h.run_horizon(tui.app, tasks_path, agents)

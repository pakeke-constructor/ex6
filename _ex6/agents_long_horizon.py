


from _ex6.provider_openai import invoke_llm as invoke_llm_openai
from _ex6.provider_anthropic import invoke_llm as invoke_llm_anthropic
from _ex6.models import M
from _ex6.tools import read_headers, read_body, glob, search, write_file, patch_file, read_file, read_image, ask_user_question, handoff, escalate, COMMANDLINE_TOOL, git_working_tree, explore_agent, AGENTS_MD, ENV_PROMPT, edit_file, read_audio
from _ex6.skills import load_skill
from _ex6.web_tools import websearch_agent
from _ex6.provider import cache_manually
from _ex6.agents import SONNET, OPUS, LUNA, TERRA, SOL, ASTRA
from _ex6.commands import format_context
import ex6
from ex6 import Context, Message
import time
import math
import os
import functools
import shutil
import threading
import git



# Task file: markdown, one `## ` heading per task (ignored inside ``` fences).
# Each task is the raw text, heading line included.

def split_tasks(text):
    preamble, tasks, in_fence = "", [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            tasks.append(line)
        elif tasks:
            tasks[-1] += line
        else:
            preamble += line
    return preamble, tasks


def join_tasks(preamble, tasks):
    return preamble + "".join(tasks)




TASK_SYSTEM_PROMPT = ex6.Message(
role ="system",
overview="main-system",
content="""\
You are a long-running coding agent who writes simple, minimal code.

<goal>
Solve the task with minimal bloat.
Don't be narrow-minded: Consider the overarching system, and consider the APIs used.
Prefer direct implementation path.
</goal>

<agent_strategy>
- Firstly, Understand request, constraints, and intent of the task.
- Scope out the relevant files, and learn about the existing systems.
- IF YOU GET STUCK: Consider the *intention* of the task, and how it affects the end user / api consumers. You are free to relax / change the requirements if the problem is particularly difficult, or hacky.
</agent_strategy>

<agent_tactics>
- Try the simplest approach first. Don't overthink.
- Tool call(s) to verify, then act. Don't read the whole codebase before a 2-line edit.
- If a search returns what you need, stop searching. Don't keep exploring "just in case."
- If your approach is blocked, don't brute force. Step back, try a different angle, or ask.
- Avoid backwards-compatibility hacks. If something is unused, delete it.

<workspace>
The `.workspace/**` folder is a list of markdown files, representing plans.

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





BASE = "master"

MAIN_PROMPT = Message(role="system", content="""\
You are running unattended; no human will answer questions.
Make reasonable decisions and note them in your final message.
Put ALL temp files, smoke tests, screenshots and outputs in `workspace/`. Never delete anything in `workspace/`.
After you are finished, your workspace/** folder will be saved for auditability purposes; this way, (your human can help you more in the future).
In the meantime, 
... blah blah todo.
""")



REVIEW_PROMPT = Message(role="system", content="""\
You are a review-agent specialized in finding simpler / better solutions, and ensuring clean code for tasks.
You review stuff at quite a high-level; the agent will be smart enough to 

# Main failures you must check for, step by step:

## Statefulness:
- Firstly, try to avoid state if possible. Immutable data is very robust.
- Otherwise, if the state can't be removed, try make it a single-source-of-truth (SSOT).
- Otherwise, if state must be duplicated, then make sure the state is either short-lived or recomputed frequently.
- Lastly, if the bad state can't be short-lived or recomputed, think about a way to invalidate it, or make the consumers aware of the duplication.
Don't forget the bigger picture. 

## Conciseness:
Every token added is an extra token that must be read in the future.
(A simple defensive if branch may seem harmless; but in a piece of code that is read frequently by hundreds of future agents, the cost adds up in terms of context window bloat.)
You should check to see for excessive comments, redundant error-checking, or any other places where the code can be shortened/simplfied whilst retaining readability.

## Code reuse and architecture:
Look at new function and new piece of code that was created.
Is all of this code placed in the right place, or should it be restructured to be elsewhere?
Are there existing functions or existing 

## Simplicity:
Take a step back, and check for a simpler solution.
If lot of code was added/changed, take a step back and evaluate the actual problem.
If the new code seems hacky, look at callers/users of the system, and reason about the intention of the system; maybe something else can change, or the requirements can be relaxed.

And finally; check that the task is actually done, and test for any obvious errors.
Give short feedback at a high level. The agent will be smart enough to interpret your high-level guidance, you SHOULD NOT write the exact code for them.
Final line exactly `VERDICT: OK` or `VERDICT: CHANGES`.
""")



DOC_PROMPT = Message(role="system", content="""\
You maintain AGENTS.md. Given a task and its diff (`git diff master`), update AGENTS.md only if a future agent needs to know something new (new module, convention, gotcha). Keep it short. If nothing to add, change nothing.

""")



# ctx.name -> list of dumped context windows for the current task
_windows: dict[str, list[str]] = {}


@functools.wraps(handoff)
def _handoff(ctx: Context, txt: str, ready: bool = False) -> str:
    if ready:
        _windows.setdefault(ctx.name, []).append(format_context(ctx))
    return handoff(ctx, txt, ready)


READ_TOOLS = [glob, search, read_file, read_headers, read_body, COMMANDLINE_TOOL, git_working_tree]
H_MAIN_TOOLS = [_handoff if t is handoff else t for t in MAIN_TOOLS]


def run(ctx, prompt):
    """Invoke ctx and block until it finishes. Returns final assistant text."""
    ctx.invoke(prompt)
    idle = 0
    while True:
        time.sleep(0.5)
        if ctx.is_running():
            idle = 0
            continue
        last = ctx.get_messages()[-1]
        if last.role == "assistant" and not last.tool_calls:
            return last.content
        # Not running and not done: either the handoff re-invoke gap, or stopped/errored.
        idle += 1
        if idle > 6:
            raise RuntimeError(f"{ctx.name} stopped without finishing")


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()



def pop_task(path):
    preamble, tasks = split_tasks(read_text(path))
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(join_tasks(preamble, tasks[1:]))
    os.replace(path + ".tmp", path)



def make_new_git_branch(repo):
    if repo.is_dirty(untracked_files=True):
        raise RuntimeError("working tree dirty")
    branch = f"horizon/{time.strftime('%Y%m%d_%H%M%S')}"
    repo.git.checkout(BASE)
    repo.git.checkout("-b", branch)
    return branch


def do_task(main_agent, preamble, t):
    run(main_agent, f"{preamble}\n{t}")


def review(repo, review_agent, t):
    """Returns feedback if changes are needed, else None."""
    repo.git.add(A=True) # so new files show in `git diff master`
    verdict = run(review_agent, f"Review the changes (`git diff master`) for this task:\n\n{t}")
    return verdict if "VERDICT: CHANGES" in verdict else None


def fix_review(main_agent, feedback):
    run(main_agent, f"Reviewer feedback:\n\n{feedback}\n\nAddress this feedback only, then stop.")


def update_docs(repo, doc_agent, t):
    repo.git.add(A=True)
    run(doc_agent, f"Task that was just completed:\n\n{t}")


def finalize_task(repo, branch, tasks_path, t, agents):
    title = t.splitlines()[0][3:].strip()
    repo.git.add(A=True)
    repo.git.commit("--allow-empty", m=f"horizon: {title}")
    repo.git.checkout(BASE)
    repo.git.merge(branch, "--no-ff", m=f"horizon merge: {title}")
    pop_task(tasks_path)

    audit = os.path.join("ex6.history", f"{time.strftime('%Y%m%d_%H%M%S')}_{repo.head.commit.hexsha[:7]}")
    os.makedirs(audit)
    if os.path.isdir("workspace"):
        shutil.move("workspace", os.path.join(audit, "workspace"))
    for ctx in agents:
        _windows.setdefault(ctx.name, []).append(format_context(ctx))
    for name, dumps in _windows.items():
        for i, d in enumerate(dumps):
            with open(os.path.join(audit, f"{name}.{i}.txt"), "w", encoding="utf-8") as f:
                f.write(d)
    with open(os.path.join(audit, "task.md"), "w", encoding="utf-8") as f:
        f.write(t)
    repo.git.add("ex6.history")
    repo.git.commit(m=f"horizon audit: {title}")

    _windows.clear()
    for ctx in agents:
        ctx.clear()
        ctx.purge_cache() # reload AGENTS.md etc


def horizon_loop(tasks_path, main_agent, review_agent, doc_agent):
    repo = git.Repo()
    while True:
        preamble, tasks = split_tasks(read_text(tasks_path))
        if not tasks: return
        t = tasks[0]

        branch = make_new_git_branch(repo)
        do_task(main_agent, preamble, t)
        feedback = review(repo, review_agent, t)
        if feedback:
            fix_review(main_agent, feedback)
        update_docs(repo, doc_agent, t)
        finalize_task(repo, branch, tasks_path, t, [main_agent, review_agent, doc_agent])


def run_horizon(app, tasks_path, main_preset=SOL, review_preset=SOL, doc_preset=SOL):
    tui = app.tui
    specs = [
        ("main_agent", main_preset, [TASK_SYSTEM_PROMPT.with_tools(H_MAIN_TOOLS), MAIN_PROMPT]),
        ("review_agent", review_preset, [REVIEW_PROMPT.with_tools(READ_TOOLS)]),
        ("doc_agent", doc_preset, [DOC_PROMPT.with_tools(READ_TOOLS + [write_file, edit_file])]),
    ]
    agents = []
    for name, preset, msgs in specs:
        old = tui.get_context(name)
        if old: tui.remove_context(old)
        ctx = Context(app, **{**preset, "name": name}, messages=msgs + [ENV_PROMPT, AGENTS_MD], yolo=True)
        tui.add_context(ctx)
        agents.append(ctx)

    def loop():
        try:
            horizon_loop(tasks_path, *agents)
        except Exception as e:
            app.debug_print(f"horizon stopped: {e}")

    threading.Thread(target=loop, daemon=True).start()



@ex6.command
def horizon(tui, tasks_path: str):
    'Run overnight task loop over a markdown task file.'
    if not tasks_path: return
    run_horizon(tui.app, os.path.abspath(tasks_path))





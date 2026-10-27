# Dependencies: GitPython, plus the imported plugins below.
# Install: python -m pip install GitPython
# Also install dependencies documented in agents.py, tools.py, skills.py, and commands.py.
# External: Git.




from _ex6.tools import handoff, AGENTS_MD, ENV_PROMPT
from _ex6.skills import load_skill, refresh_skill_doc
from _ex6.agents import OPUS, READ, WRITE, SHELL, SUBAGENTS, context_usage_reminder
from _ex6.commands import format_context
import ex6
from ex6 import Context, Message
import time
import os
import functools
import shutil
import threading
import traceback
import subprocess
import re
import git
import pyperclip



# Task file: markdown, one `##` heading per task (ignored inside ``` fences); title optional.
# Each task is the raw text, heading line included.

def split_tasks(text):
    preamble, tasks, in_fence = "", [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and (line.startswith("## ") or line.rstrip() == "##"):
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
- Break down the problem as much as you can, and try to optimize as much as you can for simplicity; less code is better.
- (Note: you can clean your context with the `handoff()` tool; don't feel pressured to complete the task in one go.)
- Then, slowly implement.

Generally, for big tasks, it should consist of a few phases:
PHASE 1: Plan. Explore codebase, map out problem-space. Write some brief high-level ideas. Then, use handoff() tool for PHASE-2.
PHASE 2: Implement. Code the solution, make changes. If you run into big issues, feel free to flip back to PHASE-1 plan-mode
PHASE 3: Smoke test. If you need, write a file into .workspace, and use it to test the stuff you just worked on.

<engineering_judgement>
- IF YOU GET STUCK: Consider the *intention* of the task, and how it affects the end user / api consumers. You are free to relax / change the requirements if the problem is particularly difficult, or hacky.
- If you see an obvious improvement to the codebase, e.g. removing an unused/orphaned function, or if you spot an obvious refactor that is very clearly beneficial: Do it.
</engineering_judgement>
</agent_strategy>

<agent_tactics>
- Try the simplest approach first. Don't overthink.
- Tool call(s) to verify, then act. Don't read the whole codebase before a 2-line edit.
- If a search returns what you need, stop searching. Don't keep exploring "just in case."
- If your approach is blocked, don't brute force. Step back, try a different angle.
- Avoid backwards-compatibility hacks. If something is unused, delete it.

<workspace>
The `.workspace/**` folder is your scratch space for this task.

Put ALL temp files, smoke tests, screenshots and outputs in `.workspace/`.
Don't delete anything in `.workspace/`; it is archived after the task so a human can audit your work / help you in the future.
`.workspace/` is gitignored, so `glob`/`search` may skip it; use exact paths to read your own files back.
</workspace>
</agent_tactics>


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









BASE = "master"

MAIN_PROMPT = Message(role="system", content="""\
You are running unattended; no human will answer questions.
Make reasonable decisions and note them in your final message.

""")




REVIEW_RESULT_KEY = "long_horizon:review_result"


def review_diff(ctx: Context, lgtm: bool = False, feedback: str = "") -> str:
    """Submit the review result. Set lgtm=true to approve; otherwise provide concise feedback describing required changes."""
    if not lgtm and not feedback.strip():
        raise ValueError("feedback is required when lgtm is false")
    ctx.data_volatile[REVIEW_RESULT_KEY] = None if lgtm else feedback.strip()
    return "Review submitted."

REVIEW_PROMPT = Message(role="system", content="""\
You are a review-agent specialized in finding simpler / better solutions, and ensuring clean code for tasks.
You review stuff at quite a high-level; the agent will be smart enough to work out the details.

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
A common failure mode of coding agents is to put a function in the files that it is familiar with, instead of placing it somewhere that makes sense.
Part of your job is guarding against that; if a new API is exposed, you should do some quick checks to see if there's a smarter place to put it.

## Simplicity:
Take a step back, and check for a simpler solution.
If lot of code was added/changed, take a step back and evaluate the actual problem.
If the new code seems hacky, look at callers/users of the system, and reason about the intention of the system;
maybe something else can change, or the requirements can be relaxed.

And finally; check that the task is actually done, and test for any obvious errors.
<agent_strategy>
- Review git diff.
- Do a deep exploration of surrounding systems.
Give short feedback at a high level. The agent will be smart enough to interpret your high-level guidance, you SHOULD NOT write the exact code for them.
- Conclude by calling `review_diff` exactly once. Set `lgtm=true` if the changes are acceptable; otherwise set `lgtm=false` and provide concise feedback describing required changes.
</agent_strategy>
""")



DOC_PROMPT = Message(role="system", content="""\
You are an agent that maintains AGENTS.md.

Given a task and its git-diff (`git diff master`), update AGENTS.md only if a future agent needs to know something new (new module, convention, gotcha).
Keep it short. If nothing to add, change nothing.

It's very important to craft a good AGENTS.md, so here is a procedure you should follow:
<agent_procedure>
- Firstly, read AGENTS.md
- Secondly, analyze the git diff, and voice out loud all the candidate changes, and their details.
- Thirdly, explicitly list what should be kept, and explicitly list what should be culled. It's very important to keep AGENTS.md minimal and simple; don't overreach. It's a directory/index, not a source of truth for everything.
- Finally; if anything survived the cull, write it into AGENTS.md.

(If changes were very small, or were a bugfix that doesn't require a change to AGENTS.md, you may ignore this procedure.)
</agent_procedure>

And, final reminder: AGENTS.md is supposed to be a *high level index*, NOT a complete map of the codebase.
Don't be overly specific, and don't be too narrow minded. It's okay if you leave stuff unspecified.
""")



HISTORY_DIR = ".ex6.history"
PENDING_DIR = os.path.join(HISTORY_DIR, "_pending") # context dumps, until the audit folder is named


def dump_context(ctx):
    os.makedirs(PENDING_DIR, exist_ok=True)
    i = sum(f.startswith(ctx.name + ".") for f in os.listdir(PENDING_DIR))
    with open(os.path.join(PENDING_DIR, f"{ctx.name}.{i}.txt"), "w", encoding="utf-8") as f:
        f.write(format_context(ctx))


@functools.wraps(handoff)
def _handoff(ctx: Context, txt: str, ready: bool = False) -> str:
    if ready:
        dump_context(ctx)
    return handoff(ctx, txt, ready)


HORIZON_MAIN_TOOLS = READ + WRITE + SHELL + SUBAGENTS + [_handoff, load_skill]


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
    return branch, repo.head.commit.hexsha


def check_git_state(repo, branch, base_sha):
    """Pipelines must leave HEAD on the task branch and not touch BASE; the loop owns merging."""
    if repo.head.is_detached or repo.active_branch.name != branch:
        raise RuntimeError(f"HEAD moved off {branch}")
    if repo.commit(BASE).hexsha != base_sha:
        raise RuntimeError(f"{BASE} moved during the task")
    if not repo.is_ancestor(base_sha, "HEAD"):
        raise RuntimeError(f"{branch} no longer descends from {BASE}")


def review(repo, review_agent, t, summary):
    """Returns feedback if changes are needed, else None."""
    repo.git.add(A=True) # so new files show in `git diff master`
    missing = object()
    review_agent.data_volatile[REVIEW_RESULT_KEY] = missing
    run(review_agent, f"Review the changes (`git diff master`) for this task:\n\n{t}\n\nThe agent's final message:\n\n{summary}")
    result = review_agent.data_volatile[REVIEW_RESULT_KEY]
    if result is missing:
        raise RuntimeError("review_agent finished without calling review_diff")
    return result


def fix_review(main_agent, feedback):
    run(main_agent, f"Reviewer feedback:\n<feedback>{feedback}</feedback>\nAddress this feedback only, then stop.")


def update_docs(repo, doc_agent, t):
    repo.git.add(A=True)
    agents_md_before = read_text("AGENTS.md")
    run(doc_agent, f"Task that was just completed:\n\n<task>{t}</task>\n\nYour job: Update the AGENTS.md if neccessary.")
    if read_text("AGENTS.md") != agents_md_before:
        doc_agent.clear() # review with fresh eyes, not as the author
        run(doc_agent, """\
Another agent just edited AGENTS.md. Read the full file, then the changes: `git diff master -- AGENTS.md`.

Does this look high quality to you? Which parts look like they were written by an LLM?
Fix the added/changed lines: rewrite or delete them. Don't touch the rest. If they're already good, do nothing.
""")


def clean_task(t):
    """Drop a bare `##` heading so untitled tasks are just their description."""
    head, _, rest = t.partition("\n")
    return rest.strip() if head.strip() == "##" else t.strip()


def task_title(t):
    return t.strip().splitlines()[0].removeprefix("## ").strip()[:72]


def archive(t, tag):
    """Move this task's context dumps + .workspace into a new history folder. Leaves no _pending behind."""
    audit = os.path.join(HISTORY_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.rename(PENDING_DIR, audit)
    if os.path.exists(".workspace"):
        shutil.move(".workspace", os.path.join(audit, ".workspace"))
    with open(os.path.join(audit, "task.md"), "w", encoding="utf-8") as f:
        f.write(t)
    return audit


def fail_task(repo, t, agents, error):
    """Commit the failed attempt on its branch (for audit); the task is then retried from scratch."""
    repo.git.add(A=True)
    repo.git.commit("--allow-empty", m=f"horizon failed: {task_title(t)}\n\n{error}")
    for ctx in agents:
        dump_context(ctx) # dumps + .workspace survive; archived with the next successful attempt
        ctx.clear()
        ctx.purge_cache()


def finalize_task(repo, branch, tasks_path, t, agents):
    title = task_title(t)
    repo.git.add(A=True)
    repo.git.commit("--allow-empty", m=f"horizon: {title}")
    diff = repo.git.diff(BASE, "HEAD")
    for ctx in agents:
        dump_context(ctx)
    repo.git.checkout(BASE)
    repo.git.merge(branch, "--no-ff", m=f"horizon merge: {title}")
    pop_task(tasks_path)

    audit = archive(t, repo.head.commit.hexsha[:7])
    with open(os.path.join(audit, "changes.diff"), "w", encoding="utf-8") as f:
        f.write(diff)
    repo.git.add(A=True) # popped tasks file (if tracked)
    if repo.is_dirty():
        repo.git.commit(m=f"horizon audit: {title}")

    for ctx in agents:
        ctx.clear()
        ctx.purge_cache() # reload AGENTS.md etc
    refresh_skill_doc()

#
#
#
#
#
#
#
#
#
#
#
#
# ================================================================
#  pipelines:
#  These are high level "flows" for what actually happens.
#  It's super easy to write your own pipelines
# ================================================================
#
#
#

def default_pipeline(repo, agents, t, preamble):
    """Per-task work. May edit/stage/commit on the task branch; see check_git_state."""
    main = agents["main"]
    review_agent = agents["review"]
    doc = agents["doc"]

    summary = run(main, f"{preamble.strip()}\n\n{t}".strip())
    feedback = review(repo, review_agent, t, summary)
    if feedback:
        fix_review(main, feedback)
    update_docs(repo, doc, t)


def barebones_pipeline(repo, agents, t, preamble):
    """Same as default, but no review."""
    main = agents["main"]
    doc = agents["doc"]

    run(main, f"{preamble.strip()}\n\n{t}".strip())
    update_docs(repo, doc, t)

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

def horizon_loop(tasks_path, agents, pipeline):
    repo = git.Repo()
    failures = 0
    shutil.rmtree(PENDING_DIR, ignore_errors=True) # stale dumps from an aborted run / stray handoffs; archive() empties it after every task
    while failures < 3:
        preamble, tasks = split_tasks(read_text(tasks_path))
        if not tasks: return
        t = clean_task(tasks[0])

        branch, base_sha = make_new_git_branch(repo)
        try:
            pipeline(repo, agents, t, preamble)
            check_git_state(repo, branch, base_sha)
        except Exception:
            check_git_state(repo, branch, base_sha) # refs moved: fatal, don't commit the failure anywhere
            fail_task(repo, t, agents.values(), traceback.format_exc())
            failures += 1
            continue
        finalize_task(repo, branch, tasks_path, t, agents.values()) # errors after merge are fatal, never retried
        failures = 0
    archive(t, "failed")
    raise RuntimeError("3 failures in a row; stopping")


HORIZON_LOCK = threading.Lock() # one loop at a time; they'd share _pending, .workspace and git


def assert_horizon_valid(tasks_path):
    if HORIZON_LOCK.locked():
        raise RuntimeError("horizon already running")
    assert os.path.isfile(tasks_path), f"tasks file not found: {tasks_path}"
    _, tasks = split_tasks(read_text(tasks_path))
    assert tasks, f"no `##` tasks in {tasks_path}"
    if os.path.exists(".gitignore"):
        ignored = [l.strip().strip("/") for l in read_text(".gitignore").splitlines()]
        assert HISTORY_DIR in ignored, f"{HISTORY_DIR} must be in .gitignore"


def default_agents(app, preset=OPUS):
    """{main, review, doc} for default_pipeline. Custom review agents must keep `review_diff`; review() reads its result."""
    main = Context(app, **{**preset, "name": "main_agent"}, yolo=True, messages=[
        TASK_SYSTEM_PROMPT.with_tools(HORIZON_MAIN_TOOLS), MAIN_PROMPT, ENV_PROMPT, AGENTS_MD])
    main.after_tool_calls(context_usage_reminder)
    review = Context(app, **{**preset, "name": "review_agent"}, yolo=True, messages=[
        REVIEW_PROMPT.with_tools(READ + SHELL + [review_diff]), ENV_PROMPT, AGENTS_MD])
    doc = Context(app, **{**preset, "name": "doc_agent"}, yolo=True, messages=[
        DOC_PROMPT.with_tools(READ + SHELL + WRITE), ENV_PROMPT
    ])
    return {
        # return new dict so caller can mutate.
        "main": main,
        "review": review,
        "doc": doc
    }


def run_horizon(app, tasks_path, agents, pipeline=default_pipeline):
    """Run the task loop in a background thread. agents: dict of name -> Context; pipeline(repo, agents, t, preamble) does each task. Replaces same-named TUI contexts."""
    assert_horizon_valid(tasks_path)
    for ctx in agents.values():
        assert ctx.yolo, f"{ctx.name} must be yolo=True; nobody is around to approve tools"
    tasks_path = os.path.abspath(tasks_path)
    tui = app.tui
    for ctx in agents.values():
        old = tui.get_context(ctx.name)
        if old: tui.remove_context(old)
        tui.add_context(ctx)

    def loop():
        try:
            horizon_loop(tasks_path, agents, pipeline)
        except Exception as e:
            app.debug_print(f"horizon stopped: {e}")
        finally:
            HORIZON_LOCK.release()

    app.long_horizon = True # never reset: once a horizon run starts, assume nobody is watching
    HORIZON_LOCK.acquire() # commands run on the TUI thread, so nothing can race the locked() check above
    threading.Thread(target=loop, daemon=True).start()




def _git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8").stdout.strip()



@ex6.command
def report(tui):
    """
    Horizon report:
    Github compare-link for the most recent run of horizon commits on master. (Copied to clipboard)
    """
    remote = _git("remote", "get-url", "origin")
    repo = re.sub(r"^(git@github\.com:|https://github\.com/)|\.git$", "", remote)
    commits = [ln.split(" ", 1) for ln in _git("log", "master", "--format=%H %s").splitlines()]
    i = 0
    while i < len(commits) and not commits[i][1].startswith("horizon"):
        i += 1 # skip non-horizon commits made after the horizon run
    j = i
    while j < len(commits) and commits[j][1].startswith("horizon"):
        j += 1

    if i == len(commits):
        lines = ["No horizon commits found on master."]
    else:
        link = f"https://github.com/{repo}/compare/{commits[j][0][:10]}...{commits[i][0][:10]}"
        pyperclip.copy(link)
        tasks = [s for _, s in commits[i:j] if s.startswith("horizon:")]
        lines = [f"{len(tasks)} horizon tasks ({j - i} commits):"]
        lines += [" " + s.removeprefix("horizon:").strip() for s in reversed(tasks)]
        lines += ["", link, "(copied to clipboard. ESC to close)"]

    def draw(buf, inpt, r):
        x, y, w, h = r
        th = tui.app.theme
        panel = ex6.Region(x, y, w, min(len(lines) + 2, h))
        px, py, pw, ph = panel
        buf.fill(panel, ' ')
        buf.rect_line(panel, txt_color=th.accent)
        for k, line in enumerate(lines[:ph - 2]):
            buf.puts(px + 2, py + 1 + k, line[:pw - 4], txt_color=th.text)

    tui.ui_panel_stack.append(draw)



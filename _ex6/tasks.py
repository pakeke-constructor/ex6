
'''
tasks.py: tools for LLMs to manage tasks.


each task is a .md file inside of `.tasks/*`.
Task spec is a given by `TASKS/task_spec.md`.

Agents interact with these tasks via tools.

tool functions we want:


task_focus(id) # focuses on this task. stores in ctx.data["tasks:id"] = id
task_create(description) # returns id (base32 id:  `dc5`)

task_read(id = None) # if None, reads focused task
task_write_plan(full_plan, id=None) # if id=None, writes focused task

task_add_log(short_str, type="BLOCKER" or "PROGRESS" or "LEARNING" or "HUMAN")
# logs to focused task. 
# should be used to record progress, learnings, or blockers on tasks.
# (Auxiliary agent should use this automatically maybe...?)
# NOTE: ALL HUMAN INPUT SHOULD BE LOGGED AS A TASK

task_query_logs(query, id=None)
# if id=None, does focused task
# spins up (cheap) subagent to query the logs. eg:
# task_query_logs("have there ")

'''

import os
import time
import random
import datetime
import ex6

TASKS_DIR = ".tasks"

def _tasks_dir(ctx):
    root = ctx.cwd or os.getcwd()
    return os.path.join(root, TASKS_DIR)

def _ensure_dir(ctx):
    os.makedirs(_tasks_dir(ctx), exist_ok=True)

def _base32_id():
    chars = "abcdefghijklmnopqrstuvwxyz234567"
    return "".join(random.choice(chars) for _ in range(3))

def _task_path(ctx, task_id):
    return os.path.join(_tasks_dir(ctx), f"{task_id}.md")

def _focused_id(ctx):
    tid = ctx.data.get("tasks:id")
    if not tid:
        raise ValueError("No task focused. Use task_focus(id) first.")
    return tid

def _resolve_id(ctx, task_id):
    return task_id if task_id else _focused_id(ctx)

def _read_task(ctx, task_id):
    path = _task_path(ctx, task_id)
    if not os.path.isfile(path):
        raise ValueError(f"Task '{task_id}' not found at {path}")
    with open(path, "r") as f:
        return f.read()

def _write_task(ctx, task_id, content):
    _ensure_dir(ctx)
    with open(_task_path(ctx, task_id), "w") as f:
        f.write(content)

def _now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


TASK_TEMPLATE = """\
# TASK: {description}

{objective}
---

<plan>
(no plan yet)
</plan>

<done_criteria>
(none yet)
</done_criteria>

<log>
[{ts}] [CREATED] {first_line}
</log>


<meta>
status: open
created_at: {ts}
</meta>
"""


def task_focus(ctx: ex6.Context, id: str) -> str:
    """
    Focus on a task. Stores in ctx so subsequent task tools default to it.
    (You should ALWAYS focus a task before working on it. This makes subsequent calls easier.)
    """
    path = _task_path(ctx, id)
    if not os.path.isfile(path):
        raise ValueError(f"Task '{id}' not found at {path}")
    ctx.data["tasks:id"] = id
    return f"Focused on task '{id}'."


def task_create(ctx: ex6.Context, description: str, objective: str = "") -> str:
    """Create a new task. Returns the task id (short base32 like 'dc5').
    description: short title for the task.
    objective: the broader goal and WHY this task matters. Human-editable."""
    _ensure_dir(ctx)
    for _ in range(20):
        tid = _base32_id()
        if not os.path.isfile(_task_path(ctx, tid)):
            break
    else:
        raise ValueError("Could not generate unique task id")
    ts = _now()
    first_line = description.split('\n')[0]
    obj = objective or "(no objective yet — edit this section to describe the goal and why)"
    content = TASK_TEMPLATE.format(description=first_line, objective=obj, first_line=first_line, ts=ts)
    _write_task(ctx, tid, content)
    ctx.data["tasks:id"] = tid
    return f"Created task '{tid}'. Auto-focused."


def task_read(ctx: ex6.Context, id: str = None) -> str:
    """Read a task's full contents. If id=None, reads the focused task."""
    tid = _resolve_id(ctx, id)
    content = _read_task(ctx, tid)
    # Label the objective section between title and --- as "task context"
    import re
    content = re.sub(
        r"(# TASK: .+\n)\n(.+?)---",
        r"\1\n<task_context>\n\2</task_context>\n---",
        content, count=1, flags=re.DOTALL
    )
    return content


def task_write_plan(ctx: ex6.Context, full_plan: str, id: str = None) -> str:
    """Overwrite the <plan> section of a task. If id=None, writes to focused task."""
    tid = _resolve_id(ctx, id)
    content = _read_task(ctx, tid)
    import re
    new_content = re.sub(
        r"<plan>.*?</plan>",
        f"<plan>\n{full_plan}\n</plan>",
        content, count=1, flags=re.DOTALL
    )
    if new_content == content:
        raise ValueError(f"No <plan> section found in task '{tid}'")
    _write_task(ctx, tid, new_content)
    return f"Updated plan for task '{tid}'."


def task_write_done_criteria(ctx: ex6.Context, criteria: str, id: str = None) -> str:
    """Overwrite the <done_criteria> section of a task. If id=None, writes to focused task.
    criteria: a short list of verifiable conditions that define when this task is complete."""
    tid = _resolve_id(ctx, id)
    content = _read_task(ctx, tid)
    import re
    new_content = re.sub(
        r"<done_criteria>.*?</done_criteria>",
        f"<done_criteria>\n{criteria}\n</done_criteria>",
        content, count=1, flags=re.DOTALL
    )
    if new_content == content:
        raise ValueError(f"No <done_criteria> section found in task '{tid}'")
    _write_task(ctx, tid, new_content)
    return f"Updated done_criteria for task '{tid}'."


def task_add_log(ctx: ex6.Context, short_str: str, type: str = "PROGRESS") -> str:
    """Add a short log entry to the focused task. Keep entries terse — a few words, not sentences.
    type: BLOCKER, PROGRESS, LEARNING, or HUMAN.
    Examples:
      task_add_log("edit_file approach fails, need write_file", "BLOCKER")
      task_add_log("auth module uses JWT not sessions", "LEARNING")
      task_add_log("refactored parse_config, tests pass", "PROGRESS")
      task_add_log("user wants retry logic on 429s", "HUMAN")"""
    tid = _focused_id(ctx)
    content = _read_task(ctx, tid)
    entry = f"[{_now()}] [{type}] {short_str}"
    new_content = content.replace("</log>", f"{entry}\n</log>")
    _write_task(ctx, tid, new_content)
    return f"Logged to task '{tid}'."


def task_close(ctx: ex6.Context, id: str = None) -> str:
    """Close a task by deleting its file. Task files are version-controlled, so nothing is lost.
    If id=None, closes the focused task."""
    tid = _resolve_id(ctx, id)
    path = _task_path(ctx, tid)
    os.remove(path)
    if ctx.data.get("tasks:id") == tid:
        del ctx.data["tasks:id"]
    return f"Closed task '{tid}' (file deleted)."


QUERY_MODEL = "google/gemini-3.1-flash-lite-preview"

QUERY_SYSTEM_PROMPT = ex6.Message(role="system", overview="task-log-query", content="""\
You answer questions about a task's log. You receive the full task file and a question.
Be EXTREMELY concise. Grammar doesn't matter. Just the facts.
If the answer is one word, say one word. No fluff, no preamble, no hedging.
""")


def task_query_logs(ctx: ex6.Context, question: str, id: str = None) -> str:
    """Ask a question about a task's logs. Spins up a cheap subagent to answer.
    question: a natural-language question, e.g. "any blockers?" or "what was the last finding?"
    If id=None, queries the focused task."""
    tid = _resolve_id(ctx, id)
    task_content = _read_task(ctx, tid)
    sub = ex6.Context("task_query", model=QUERY_MODEL, messages=[QUERY_SYSTEM_PROMPT], reasoning="none")
    sub.parent = ctx.name
    prompt = f"<task>\n{task_content}\n</task>\n\nQuestion: {question}"
    sub.invoke(prompt)
    while sub.llm_is_running:
        time.sleep(0.05)
    result = sub.messages[-1].content if sub.messages else "No answer."
    del ex6.state.contexts[sub.name]
    return result


def task_list(ctx: ex6.Context) -> str:
    """List all tasks with their id, status, and title."""
    _ensure_dir(ctx)
    import re
    td = _tasks_dir(ctx)
    files = sorted(f for f in os.listdir(td) if f.endswith(".md"))
    if not files:
        return "No tasks."
    lines = []
    focused = ctx.data.get("tasks:id")
    for f in files:
        tid = f.removesuffix(".md")
        with open(os.path.join(td, f)) as fh:
            text = fh.read()
        title_m = re.search(r"^# TASK: (.+)$", text, re.MULTILINE)
        title = title_m.group(1) if title_m else "?"
        status_m = re.search(r"^status: (.+)$", text, re.MULTILINE)
        status = status_m.group(1).strip() if status_m else "?"
        marker = " *" if tid == focused else ""
        lines.append(f"[{tid}] ({status}) {title}{marker}")
    return "\n".join(lines)

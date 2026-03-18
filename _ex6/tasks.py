
"""
tasks.py: tools for LLMs to manage tasks.

Each task is a .md file inside of `.tasks/`.
Task spec is given by `TASKS/task_spec.md`.
Agents interact with these tasks via tools.
"""

import os
import time
import random
import datetime
import ex6

TASKS_DIR = ".tasks"

def _ensure_dir():
    os.makedirs(TASKS_DIR, exist_ok=True)

def _base32_id():
    chars = "abcdefghijklmnopqrstuvwxyz234567"
    return "".join(random.choice(chars) for _ in range(3))

def _task_path(task_id):
    return os.path.join(TASKS_DIR, f"{task_id}.md")

def _focused_id(ctx):
    tid = ctx.data.get("tasks:id")
    if not tid:
        raise ValueError("No task focused. Use task_focus(id) first.")
    return tid

def _resolve_id(ctx, task_id):
    return task_id if task_id else _focused_id(ctx)

def _read_task(task_id):
    path = _task_path(task_id)
    if not os.path.isfile(path):
        raise ValueError(f"Task '{task_id}' not found at {path}")
    with open(path, "r") as f:
        return f.read()

def _write_task(task_id, content):
    _ensure_dir()
    with open(_task_path(task_id), "w") as f:
        f.write(content)

def _now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


TASK_TEMPLATE = """\
# TASK: {description}

<plan>
(no plan yet)
</plan>


<log>
[{ts}] [CREATED] {description}
</log>


<meta>
status: open
created_at: {ts}
</meta>
"""


def task_focus(ctx: ex6.Context, id: str) -> str:
    """Focus on a task. Stores in ctx so subsequent task tools default to it."""
    path = _task_path(id)
    if not os.path.isfile(path):
        raise ValueError(f"Task '{id}' not found at {path}")
    ctx.data["tasks:id"] = id
    return f"Focused on task '{id}'."


def task_create(ctx: ex6.Context, description: str) -> str:
    """Create a new task. Returns the task id (short base32 like 'dc5')."""
    _ensure_dir()
    for _ in range(20):
        tid = _base32_id()
        if not os.path.isfile(_task_path(tid)):
            break
    else:
        raise ValueError("Could not generate unique task id")
    ts = _now()
    content = TASK_TEMPLATE.format(description=description, ts=ts)
    _write_task(tid, content)
    ctx.data["tasks:id"] = tid
    return f"Created task '{tid}'. Auto-focused."


def task_read(ctx: ex6.Context, id: str = None) -> str:
    """Read a task's full contents. If id=None, reads the focused task."""
    tid = _resolve_id(ctx, id)
    return _read_task(tid)


def task_write_plan(ctx: ex6.Context, full_plan: str, id: str = None) -> str:
    """Overwrite the <plan> section of a task. If id=None, writes to focused task."""
    tid = _resolve_id(ctx, id)
    content = _read_task(tid)
    import re
    new_content = re.sub(
        r"<plan>.*?</plan>",
        f"<plan>\n{full_plan}\n</plan>",
        content, count=1, flags=re.DOTALL
    )
    if new_content == content:
        raise ValueError(f"No <plan> section found in task '{tid}'")
    _write_task(tid, new_content)
    return f"Updated plan for task '{tid}'."


def task_add_log(ctx: ex6.Context, short_str: str, type: str = "PROGRESS") -> str:
    """Add a log entry to the focused task.
    type: BLOCKER, PROGRESS, LEARNING, or HUMAN.
    Use to record progress, learnings, blockers, or human input."""
    tid = _focused_id(ctx)
    content = _read_task(tid)
    entry = f"[{_now()}] [{type}] {short_str}"
    new_content = content.replace("</log>", f"{entry}\n</log>")
    _write_task(tid, new_content)
    return f"Logged to task '{tid}'."


def task_list(ctx: ex6.Context) -> str:
    """List all tasks with their id, status, and title."""
    _ensure_dir()
    import re
    files = sorted(f for f in os.listdir(TASKS_DIR) if f.endswith(".md"))
    if not files:
        return "No tasks."
    lines = []
    focused = ctx.data.get("tasks:id")
    for f in files:
        tid = f.removesuffix(".md")
        with open(os.path.join(TASKS_DIR, f)) as fh:
            text = fh.read()
        title_m = re.search(r"^# TASK: (.+)$", text, re.MULTILINE)
        title = title_m.group(1) if title_m else "?"
        status_m = re.search(r"^status: (.+)$", text, re.MULTILINE)
        status = status_m.group(1).strip() if status_m else "?"
        marker = " *" if tid == focused else ""
        lines.append(f"[{tid}] ({status}) {title}{marker}")
    return "\n".join(lines)

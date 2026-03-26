'''
plan.py: simple plan files for agents.

Each plan is a .md file in `.plans/`.
Agents write freeform content. Logs get appended to the bottom.
'''

import os
import random
import datetime
import ex6

PLANS_DIR = ".plans"

def _plans_dir(ctx):
    return os.path.join(ctx.cwd or os.getcwd(), PLANS_DIR)

def _ensure_dir(ctx):
    os.makedirs(_plans_dir(ctx), exist_ok=True)

def _base32_id():
    chars = "abcdefghijklmnopqrstuvwxyz234567"
    return "".join(random.choice(chars) for _ in range(3))

def _plan_path(ctx, plan_id):
    return os.path.join(_plans_dir(ctx), f"{plan_id}.md")

def _focused_id(ctx):
    tid = ctx.data.get("plan:id")
    if not tid:
        raise ValueError("No plan focused. Use plan_read(id) or plan_write(content) first.")
    return tid

def _resolve_and_focus(ctx, plan_id):
    tid = plan_id if plan_id else _focused_id(ctx)
    ctx.data["plan:id"] = tid
    return tid

def _now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


def plan_write(ctx: ex6.Context, content: str, id: str = None) -> str:
    """Write a plan. If id=None, creates a new plan and returns its id.
    If id is given, overwrites that plan. Auto-focuses.
    content: freeform markdown. Write whatever structure makes sense."""
    _ensure_dir(ctx)
    if id:
        tid = id
    else:
        for _ in range(20):
            tid = _base32_id()
            if not os.path.isfile(_plan_path(ctx, tid)):
                break
        else:
            raise ValueError("Could not generate unique plan id")
    with open(_plan_path(ctx, tid), "w") as f:
        f.write(content)
    ctx.data["plan:id"] = tid
    created = "Created" if not id else "Updated"
    return f"{created} plan '{tid}'. Focused."


def plan_read(ctx: ex6.Context, id: str = None) -> str:
    """Read a plan's full contents. If id=None, reads the focused plan. Auto-focuses."""
    tid = _resolve_and_focus(ctx, id)
    path = _plan_path(ctx, tid)
    if not os.path.isfile(path):
        raise ValueError(f"Plan '{tid}' not found at {path}")
    with open(path, "r") as f:
        return f.read()


def plan_add_log(ctx: ex6.Context, short_str: str, type: str = "PROGRESS") -> str:
    """Append a log entry to the focused plan.
    type: BLOCKER, PROGRESS, LEARNING, or HUMAN."""
    tid = _focused_id(ctx)
    path = _plan_path(ctx, tid)
    with open(path, "r") as f:
        content = f.read()
    entry = f"\n[{_now()}] [{type}] {short_str}"
    with open(path, "w") as f:
        f.write(content + entry)
    return f"Logged to plan '{tid}'."


def plan_done(ctx: ex6.Context, id: str = None) -> str:
    """Delete a plan file. Plans are version-controlled, so nothing is lost.
    If id=None, closes the focused plan."""
    tid = _resolve_and_focus(ctx, id)
    path = _plan_path(ctx, tid)
    if not os.path.isfile(path):
        raise ValueError(f"Plan '{tid}' not found")
    os.remove(path)
    if ctx.data.get("plan:id") == tid:
        del ctx.data["plan:id"]
    return f"Closed plan '{tid}'."


def plan_list(ctx: ex6.Context) -> str:
    """List all plans with their id and first line."""
    _ensure_dir(ctx)
    td = _plans_dir(ctx)
    files = sorted(f for f in os.listdir(td) if f.endswith(".md"))
    if not files:
        return "No plans."
    lines = []
    focused = ctx.data.get("plan:id")
    for f in files:
        pid = f.removesuffix(".md")
        with open(os.path.join(td, f)) as fh:
            first_line = fh.readline().strip() or "(empty)"
        marker = " *" if pid == focused else ""
        lines.append(f"[{pid}] {first_line}{marker}")
    return "\n".join(lines)

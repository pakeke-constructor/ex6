'''
plans.py: `/plans` lists `.plans/*.md` with age and status. `g` runs the selected plan in a new tui.favorite agent.
`/plans triage` starts a tui.favorite agent that deletes done/obsolete plans and trims half-done ones.
'''

import os
import re
import time
import subprocess
from typing import Optional
import ex6

STALE_DAYS = 14

TRIAGE_PROMPT = """Triage the plans in `.plans/`. For each plan:
- Read it. Find when it was added (`git log --diff-filter=A --format=%ad -- <file>`), then check `git log --since=<that date> --stat` and the code it names.
- Verdict: DONE (delete it), PARTIAL (rewrite it to only the remaining work and update its Status line), OBSOLETE (delete it), or KEEP (add a Status line if it's missing).
First print a short table: plan, verdict, one-line reason. Then make the edits. Plans are in git, so deleting loses nothing."""


def _git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip()


def _load_plans(cwd):
    d = os.path.join(cwd, ".plans")
    if not os.path.isdir(d):
        return []
    plans = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".md"):
            continue
        path = os.path.join(d, f)
        with open(path, encoding="utf-8") as fh:
            head = [fh.readline().strip() for _ in range(5)]
        ts = _git(cwd, "log", "-1", "--format=%ct", "--", path)
        age = int((time.time() - (int(ts) if ts else os.path.getmtime(path))) / 86400)
        status = next((l.removeprefix("Status:").strip() for l in head if l.startswith("Status:")), "?")
        flags = []
        if age > STALE_DAYS:
            flags.append("STALE")
        m = re.match(r"blocked on `?([\w.-]+?)(?:\.md)?`?(\s|$)", status)
        if m:
            blocker = os.path.join(d, m.group(1) + ".md")
            # only flag real plans: the blocker existed in git once, and is gone now (done)
            if not os.path.isfile(blocker) and _git(cwd, "log", "-1", "--format=%h", "--", blocker):
                flags.append("UNBLOCKED")
        plans.append({
            "name": f.removesuffix(".md"),
            "age": age,
            "status": status,
            "flags": flags,
            "title": head[0].lstrip("# "),
        })
    plans.sort(key=lambda p: -p["age"])
    return plans


def _spawn(tui, name, prompt):
    fav = tui.favorite or tui.current
    if tui.get_context(name):
        name = ex6._unique_name(tui, name)
    ctx = fav.fork(name)
    ctx.clear()
    ex6._add(tui, ctx)
    ctx.invoke(prompt)


def _panel(tui, plans):
    sel = [0]
    scroll = [0]
    def draw(buf, inpt, r):
        x, y, w, h = r
        th = tui.app.theme
        buf.fill(r, ' ')
        buf.rect_line(r, txt_color=th.accent)
        if inpt.consume('j', 'KEY_DOWN'): sel[0] = min(sel[0] + 1, len(plans) - 1)
        if inpt.consume('k', 'KEY_UP'): sel[0] = max(sel[0] - 1, 0)
        if inpt.consume('g'):
            p = plans[sel[0]]
            tui.ui_panel_stack.remove(draw)
            _spawn(tui, p["name"], f"Execute the plan in `.plans/{p['name']}.md`.")
            return
        buf.puts(x + 2, y + 1, "plans (oldest first)   g go  j/k select  esc close"[:w - 4], txt_color=th.accent)
        visible = h - 3
        if sel[0] < scroll[0]: scroll[0] = sel[0]
        if sel[0] >= scroll[0] + visible: scroll[0] = sel[0] - visible + 1
        nw = max(len(p["name"]) for p in plans)
        for i, p in enumerate(plans[scroll[0]:scroll[0] + visible]):
            row = y + 2 + i
            if scroll[0] + i == sel[0]:
                buf.fill_bg(x + 1, row, w - 2, 1, th.selection)
            flags = " ".join(p["flags"])
            line = f"{p['name']:<{nw}}  {p['age']:>3}d  {flags:<15}  {p['status'][:30]:<30}  {p['title']}"
            color = th.warning if p["flags"] else th.text
            buf.puts(x + 2, row, line[:w - 4], txt_color=color)
    tui.ui_panel_stack.append(draw)


@ex6.command
def plans(tui, sub: Optional[str]):
    """List .plans (g runs one in a new agent). `/plans triage` cleans up done/stale plans."""
    cwd = tui.current.cwd or os.getcwd()
    if sub == "triage":
        _spawn(tui, "plans-triage", TRIAGE_PROMPT)
        return
    found = _load_plans(cwd)
    if not found:
        tui.echo("no plans")
        return
    _panel(tui, found)

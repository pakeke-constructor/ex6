'''
plans.py: `/plans` lists `.plans/*.md` with age and status. `g` runs the selected plan in a new tui.favorite agent.
`/plans triage` starts a tui.favorite agent that deletes done/obsolete plans and trims half-done ones.
'''

import os
import re
import time
import subprocess
import textwrap
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
            text = fh.read()
        head = [l.strip() for l in text.splitlines()[:5]] or [""]
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
        name = f.removesuffix(".md")
        plans.append({
            "name": name,
            "age": age,
            "status": status,
            "flags": flags,
            "title": head[0].lstrip("# ").removeprefix(name + ":").strip(),
            "text": text,
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
    st = {"sel": 0, "top": 0, "pv": 0}
    nw = max(len(p["name"]) for p in plans)
    metas = [" ".join(p["flags"] + [p["status"]]) for p in plans]
    mw = min(28, max(len(m) for m in metas))
    def draw(buf, inpt, r):
        x, y, w, h = r
        th = tui.app.theme
        if inpt.consume('j', 'KEY_DOWN'): st.update(sel=min(st["sel"] + 1, len(plans) - 1), pv=0)
        if inpt.consume('k', 'KEY_UP'): st.update(sel=max(st["sel"] - 1, 0), pv=0)
        if inpt.consume('KEY_PGDOWN'): st["pv"] += 10
        if inpt.consume('KEY_PGUP'): st["pv"] = max(st["pv"] - 10, 0)
        p = plans[st["sel"]]
        if inpt.consume('g'):
            tui.ui_panel_stack.remove(draw)
            _spawn(tui, p["name"], f"Execute the plan in `.plans/{p['name']}.md`.")
            return
        ix, iw = x + 4, w - 8  # inner padding
        buf.fill(r, ' ')
        buf.rect_line(r, txt_color=th.accent)
        buf.puts(x + 2, y, f" plans ({len(plans)}) ", style='bold', txt_color=th.accent)
        hint = " j/k select   pgup/pgdn scroll   g run   esc close "
        buf.puts(x + w - 2 - len(hint), y, hint, txt_color=th.muted)
        lh = (h - 6) // 2  # list on top, preview below; blank row above each
        st["top"] = min(max(st["top"], st["sel"] - lh + 1), st["sel"])
        for i in range(st["top"], min(len(plans), st["top"] + lh)):
            q, on, row = plans[i], i == st["sel"], y + 2 + i - st["top"]
            bg = th.selection if on else None
            cols = [
                (f"{q['name']:<{nw}}   ", th.error, 'bold'),
                (f"{q['age']:>3}d   ", th.muted, None),
                (f"{metas[i][:mw]:<{mw}}   ", th.warning if q["flags"] else th.success, None),
                (q["title"], th.text if on else th.muted, None),
            ]
            buf.puts(x + 1, row, " " * (w - 2), bg_color=bg)
            if on: buf.puts(ix - 2, row, "▶", txt_color=th.accent, bg_color=bg)
            cx = ix
            for txt, color, style in cols:
                buf.puts(cx, row, txt[:max(0, ix + iw - cx)], style=style, txt_color=color, bg_color=bg)
                cx += len(txt)
        sep = y + 3 + lh
        buf.puts(x, sep, '├' + '─' * (w - 2) + '┤', txt_color=th.accent)
        buf.puts(x + 2, sep, f" .plans/{p['name']}.md ", style='bold', txt_color=th.error)
        lines = []
        for line in p["text"].splitlines():
            lines += textwrap.wrap(line, iw) or [""]
        st["pv"] = min(st["pv"], max(0, len(lines) - 1))
        for j, line in enumerate(lines[st["pv"]:st["pv"] + y + h - 3 - sep]):
            heading = line.startswith("#")
            buf.puts(ix, sep + 2 + j, line, style='bold' if heading else None, txt_color=th.accent_alt if heading else th.text)
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

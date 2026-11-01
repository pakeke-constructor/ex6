"""Git worktrees for agents: `attach(ctx)` gives ctx its own branch + dir (ctx.cwd).
Stored in `<repo>/.ex6.worktrees/<name>/`, branch `ex6/<name>`. Agent merges itself via shell (see prompt).
`<name>.used` marker next to each worktree: mtime = last used, content = base branch."""
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
import ex6

CLEANUP_MB = 500
STALE_DAYS = 7
WORKTREE_COPY = [".env"]  # untracked files copied into new worktrees


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def _root():
    """Main checkout root (from the process cwd, not a ctx)."""
    return Path(_git(os.getcwd(), "rev-parse", "--show-toplevel").stdout.strip())


def _wt_dir():
    return _root() / ".ex6.worktrees"


def _prompt(path, branch, base, main):
    return f"""<worktree>
You are working in a git worktree at {path}, branch {branch}, based on {base}.
The user's main checkout is {main}; don't touch it, except for the final merge.
Commit as you go. When the task is done:
1. In your worktree: `git merge {base}`, resolve any conflicts there, commit.
2. Check `git -C "{main}" branch --show-current` prints {base}. If not, stop and ask the user.
3. `git -C "{main}" merge --ff-only {branch}`. If it fails, stop and tell the user why.
</worktree>"""


def attach(ctx: ex6.Context, base=None) -> str:
    """Give ctx a new worktree + branch. Call before invoking (or on an idle ctx)."""
    root = _root()
    base = base or _git(root, "branch", "--show-current").stdout.strip()
    if not base:
        raise RuntimeError("main checkout has no current branch; pass base=")
    slug = re.sub(r"[^\w-]+", "-", ctx.name).strip("-").lower() or "agent"
    name = f"{slug}-{secrets.token_hex(2)}"
    branch = f"ex6/{name}"
    path = root / ".ex6.worktrees" / name
    r = _git(root, "worktree", "add", "-b", branch, str(path), base)
    if r.returncode:
        raise RuntimeError(r.stderr.strip())
    for f in WORKTREE_COPY:
        if (root / f).is_file():
            shutil.copy2(root / f, path / f)
    (path.parent / f"{name}.used").write_text(base, encoding="utf-8")

    ctx.cwd = str(path)
    ctx.data["worktree:name"] = name
    ctx.data["worktree:base"] = base
    msg = ex6.Message(role="system", content=_prompt(path, branch, base, root), overview="worktree")

    def edit(messages):
        messages[:] = [m for m in messages if m.overview != "worktree"]  # re-attach (eg a fork) replaces it
        n_sys = 0
        while n_sys < len(messages) and messages[n_sys].role == "system":
            n_sys += 1
        messages.insert(n_sys, msg)
    ctx.edit_messages(edit)
    ctx.purge_cache()  # env prompt etc. were snapshotted with the old cwd
    return str(path)


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def _live_cwds(app):
    tui = app.tui
    return {_norm(c.cwd) for c in (tui.contexts if tui else []) if c.cwd}


def _info(path):
    name = path.name
    marker = path.parent / f"{name}.used"
    base = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    root = _root()
    merged = bool(base) and _git(root, "merge-base", "--is-ancestor", f"ex6/{name}", base).returncode == 0
    dirty = bool(_git(path, "status", "--porcelain").stdout.strip())
    used = marker.stat().st_mtime if marker.exists() else path.stat().st_mtime
    return {"name": name, "base": base, "merged": merged, "dirty": dirty, "used": used}


def _size(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def _remove(path, info, app):
    root = _root()
    if info["dirty"]:
        _git(path, "add", "-A")
        _git(path, "commit", "-m", "ex6: wip (auto-cleanup)")
    r = _git(root, "worktree", "remove", "--force", str(path))
    if r.returncode:
        app.debug_print(f"[worktrees] remove {path.name} failed: {r.stderr.strip()}")
        return
    if info["merged"]:  # unmerged branches stay: work is recoverable
        _git(root, "branch", "-D", f"ex6/{path.name}")
    (path.parent / f"{path.name}.used").unlink(missing_ok=True)
    app.debug_print(f"[worktrees] removed {path.name}")


def cleanup(app):
    """Over CLEANUP_MB: remove stale worktrees (oldest first) not used by a live context."""
    folder = _wt_dir()
    if not folder.is_dir():
        return
    _git(_root(), "worktree", "prune")
    paths = [p for p in folder.iterdir() if p.is_dir()]
    sizes = {p: _size(p) for p in paths}
    total = sum(sizes.values())
    live = _live_cwds(app)
    infos = sorted((_info(p) | {"path": p} for p in paths), key=lambda i: i["used"])
    for info in infos:
        if total <= CLEANUP_MB * 1e6:
            break
        p = info["path"]
        if time.time() - info["used"] < STALE_DAYS * 86400 or _norm(p) in live:
            continue
        _remove(p, info, app)
        total -= sizes[p]


@ex6.after_llm_turn
def _touch(ctx):
    name = ctx.data.get("worktree:name")
    if name:
        (_wt_dir() / f"{name}.used").touch()


@ex6.context_removed
def _on_removed(ctx):
    name = ctx.data.get("worktree:name")
    if not name or _norm(ctx.cwd) in _live_cwds(ctx.app):
        return
    path = _wt_dir() / name
    if not path.is_dir():
        return
    info = _info(path)
    if info["merged"] and not info["dirty"]:
        _remove(path, info, ctx.app)


@ex6.command
def wt(tui, base: Optional[str]):
    'Attach current agent to a new git worktree (optional base branch).'
    ctx = tui.current
    if not ctx or ctx.is_running():
        tui.echo("no idle agent selected", error=True)
        return
    try:
        tui.echo(f"worktree: {attach(ctx, base)}")
    except Exception as e:
        tui.echo(f"worktree failed: {e}", error=True)


@ex6.command
def wts(tui):
    'List ex6 git worktrees.'
    folder = _wt_dir()
    paths = [p for p in folder.iterdir() if p.is_dir()] if folder.is_dir() else []
    live = _live_cwds(tui.app)
    lines = [f"{len(paths)} worktrees in {folder}", ""]
    for p in paths:
        i = _info(p)
        days = (time.time() - i["used"]) / 86400
        flags = ("merged " if i["merged"] else "") + ("dirty " if i["dirty"] else "") + ("live" if _norm(p) in live else "")
        lines.append(f"{i['name']:<32} base={i['base']:<12} {days:5.1f}d {_size(p) / 1e6:7.1f}MB  {flags}")
    ex6.text_panel(tui, lines)


def setup(app):
    threading.Thread(target=cleanup, args=(app,), daemon=True).start()

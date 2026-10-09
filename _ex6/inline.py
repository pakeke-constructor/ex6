"""
Inline file prompts: write a line ending in ;;; to launch a coding agent.
Uses watchfiles for event-driven FS watching, Git for change detection.

Usage from agents.py or similar:
    from _ex6.inline import setup as inline_setup
    ctx = Context(app, "inline_template", model=..., reasoning=..., messages=[...], invoke_llm=...)
    inline_setup(app, ctx)
"""

import threading
import subprocess
import os
import re
from pathlib import Path

import ex6
from ex6 import Context

MARKER = ";;;"
DEBOUNCE_MS = 500

_counter = 0
_counter_lock = threading.Lock()


def _next_id():
    global _counter
    with _counter_lock:
        _counter += 1
        return _counter


def _git_changed_lines(root: Path, filepath: Path) -> set[int] | None:
    """Return set of current-file line numbers that differ from HEAD.
    Returns None if file is untracked."""
    rel = filepath.relative_to(root).as_posix()

    r = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=str(root), capture_output=True, timeout=10
    )
    if r.returncode != 0:
        return None  # untracked

    r = subprocess.run(
        ["git", "diff", "HEAD", "--unified=0", "--no-color", "--no-ext-diff", "--", rel],
        cwd=str(root), capture_output=True, text=True, timeout=10
    )
    if r.returncode != 0:
        return None

    changed = set()
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", r.stdout, re.MULTILINE):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) else 1
        for i in range(start, start + count):
            changed.add(i)
    return changed


def _extract_prompts(lines: list[str], changed_lines: set[int] | None, is_tracked: bool):
    """Extract prompt regions ending with ;;; marker.
    Tracked: collects contiguous changed lines around marker.
    Untracked: only the marker line itself.
    """
    prompts = []
    used = set()

    for i, line in enumerate(lines):
        ln = i + 1
        if ln in used:
            continue
        if not line.rstrip('\r\n').endswith(MARKER):
            continue

        if is_tracked and changed_lines is not None:
            if ln not in changed_lines:
                continue

            start = ln
            while (start - 1) in changed_lines and (start - 1) not in used:
                start -= 1
            end = ln
            while (end + 1) in changed_lines and (end + 1) not in used and end < len(lines):
                end += 1

            region = []
            for n in range(start, end + 1):
                region.append(lines[n - 1])
                used.add(n)

            region[-1] = region[-1].rstrip('\r\n')[:-len(MARKER)]
            text = "".join(region).strip()
            if text:
                prompts.append((start, end, text))
        else:
            used.add(ln)
            text = line.rstrip('\r\n')[:-len(MARKER)].strip()
            if text:
                prompts.append((ln, ln, text))

    return prompts


def _process_file(app, root: Path, template_ctx: Context, filepath: Path):
    try:
        raw = filepath.read_bytes()
    except OSError:
        return

    if MARKER.encode() not in raw:
        return
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return

    lines = text.splitlines(keepends=True)
    if not any(l.rstrip('\r\n').endswith(MARKER) for l in lines):
        return

    changed_lines = _git_changed_lines(root, filepath)
    is_tracked = changed_lines is not None
    if is_tracked and not changed_lines:
        return

    prompts = _extract_prompts(lines, changed_lines, is_tracked)
    if not prompts:
        return

    try:
        if filepath.read_bytes() != raw:
            return
    except OSError:
        return

    new_lines = list(lines)
    for start, end, _ in reversed(prompts):
        del new_lines[start - 1:end]

    try:
        filepath.write_bytes("".join(new_lines).encode("utf-8"))
    except OSError:
        return

    tui = app.tui
    if not tui:
        return

    rel_path = filepath.relative_to(root).as_posix()
    for start, _, prompt_text in prompts:
        n = _next_id()
        name = f"inline_{Path(rel_path).stem}_{start}_{n}"
        ctx = template_ctx.fork(name)
        tui.add_context(ctx)
        ctx.invoke(f"({rel_path}, line {start})\n{prompt_text}")


def _watcher_loop(app, root: Path, template_ctx: Context, stop_event: threading.Event):
    from watchfiles import watch, Change, DefaultFilter

    for changes in watch(
        str(root),
        watch_filter=DefaultFilter(),
        stop_event=stop_event,
        debounce=DEBOUNCE_MS,
        step=100,
        raise_interrupt=False,
    ):
        if stop_event.is_set():
            break

        paths = set()
        for change_type, path in changes:
            if change_type != Change.deleted:
                p = Path(path).resolve()
                if p.is_file():
                    paths.add(p)

        for p in paths:
            try:
                _process_file(app, root, template_ctx, p)
            except Exception as e:
                app.debug_print(f"[inline] error processing {p}: {e}")


def setup_inline_watcher(app, ctx: Context):
    """Start inline watcher. ctx is a template Context — each inline prompt forks from it."""
    if not app.tui:
        return

    root = Path(os.getcwd()).resolve()
    r = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(root), capture_output=True, text=True, timeout=5
    )
    if r.returncode != 0:
        app.debug_print("[inline] not in a git work tree, disabled")
        return

    git_root = Path(r.stdout.strip()).resolve()

    key = "inline:stop_event"
    if key in app.plugin_data:
        return

    stop_event = threading.Event()
    app.plugin_data[key] = stop_event

    t = threading.Thread(
        target=_watcher_loop,
        args=(app, git_root, ctx, stop_event),
        daemon=True,
        name="inline-watcher",
    )
    t.start()

    import atexit
    atexit.register(stop_event.set)
    app.debug_print(f"[inline] watching {git_root}")

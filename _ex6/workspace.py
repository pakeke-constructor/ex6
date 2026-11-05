import ex6
import os
import time
import shutil


def _newest_mtime(path):
    newest = os.path.getmtime(path)
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            newest = max(newest, os.path.getmtime(os.path.join(root, name)))
    return newest


def sweep_workspace(app, days=3, folder=".workspace"):
    """Delete top-level entries in `folder` untouched for `days` (dirs judged by their newest file)."""
    if not os.path.isdir(folder):
        return
    cutoff = time.time() - days * 86400
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if _newest_mtime(path) > cutoff:
            continue
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
        app.debug_print(f"sweep_workspace: removed {path}")

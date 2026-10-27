import os
import re

import ex6


_DUMP_NAME = re.compile(r"^\d{12}\.(?:txt|json)$")


def check_vscode_diagnostics(ctx: ex6.Context) -> str:
    """Read the newest diagnostics snapshot exported by VS Code Diagnostic Dumper."""
    directory = ctx.resolve(os.path.join(".vscode", "diagnostic-dumps"))
    if not os.path.isdir(directory):
        return "No VS Code diagnostics snapshot found. Ensure Diagnostic Dumper is active for this workspace."

    names = [name for name in os.listdir(directory) if _DUMP_NAME.fullmatch(name)]
    if not names:
        return "No VS Code diagnostics snapshot found. Ensure Diagnostic Dumper is active for this workspace."

    path = os.path.join(directory, max(names))
    with open(path, encoding="utf-8") as f:
        return f.read()

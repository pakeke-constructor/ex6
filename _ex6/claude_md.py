import os
import ex6

def _get_content(ctx):
    for p in ["CLAUDE.md", ".claude/CLAUDE.md"]:
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    return "(no CLAUDE.md found)"

CLAUDE_MD = ex6.Message(role="system", content=_get_content)

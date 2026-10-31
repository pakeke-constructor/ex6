# Prompt files: `_prompts/*.md` (core, user, project; later dirs win). `;name` in an input box pastes the body verbatim.

from pathlib import Path
import ex6
from _ex6.skills import _parse_frontmatter

PROMPT_DIRS = [Path(ex6._ex6_dir) / "_ex6" / "_prompts", ex6.get_folder() / "_ex6" / "_prompts", Path("_ex6/_prompts")]


def prompts() -> dict:
    """name -> body. Read from disk every call (files are the truth)."""
    out = {}
    for d in PROMPT_DIRS:
        for p in sorted(d.glob("*.md")):
            name, _, body = _parse_frontmatter(p, p.read_text(encoding="utf-8"))
            out[name] = body.strip()
    return out


@ex6.handler
def expand_abbrev(app, name):
    return prompts().get(name)

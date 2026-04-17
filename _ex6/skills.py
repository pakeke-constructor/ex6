"""
skills: Progressive disclosure. LLMs can list/load skill files.
Skill files live in _ex6/skills/*.md.
Requires yaml frontmatter with name and description fields.

Skill format uses anthropic's skill format, using progressive disclosure:


---
name: <skill-id>
description: <what it does + when to use it>
---
# Instructions...
blah blah.
blah
"""
import ex6
import yaml
from pathlib import Path
from typing import Optional


_skills_dir = Path.cwd() / "_ex6" / "_skills"



def _parse_frontmatter(path, text):
    """Parse yaml frontmatter from --- delimited block. Returns (name, description, body)."""
    if not text.startswith("---"):
        raise ValueError(f"{path}: missing frontmatter")
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = yaml.safe_load("\n".join(lines[1:i])) or {}
            body = "\n".join(lines[i+1:]).lstrip("\n")
            if "name" not in fm:
                raise ValueError(f"{path}: frontmatter missing 'name'")
            if "description" not in fm:
                raise ValueError(f"{path}: frontmatter missing 'description'")
            return fm["name"], fm["description"], body
    raise ValueError(f"{path}: missing closing '---'")


def _list_skills():
    if not _skills_dir.exists(): return []
    out = []
    for p in sorted(_skills_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        name, desc, _ = _parse_frontmatter(p, text)
        out.append((name, desc))
    return out


def load_skill(ctx: ex6.Context, skill_id: str = "") -> str:
    if not skill_id:
        lines = [f"{sid}: {desc}" for sid, desc in _list_skills()]
        return "\n".join(lines) if lines else "(no skills are available!)"
    path = _skills_dir / f"{skill_id}.md"
    if not path.exists():
        avail = ", ".join(sid for sid, _ in _list_skills())
        raise ValueError(f"Unknown skill '{skill_id}'. Available: {avail}")
    text = path.read_text(encoding="utf-8")
    name, _, body = _parse_frontmatter(path, text)
    ctx.messages.append(ex6.Message(role="user", content=f"[skill: {name}]\n{body}"))
    return f"Loaded skill '{name}'."



SKILL_DOC = "Load a skill's content into the conversation.\n"
_all_skills = _list_skills()
if not _all_skills:
    SKILL_DOC += "No skills exist."
else:
    SKILL_DOC += "Available skills:\n"
    for _sid, _desc in _all_skills:
        SKILL_DOC += f"  {_sid}: {_desc}\n"


load_skill.__doc__ = SKILL_DOC


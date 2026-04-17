"""
skills: Progressive disclosure. LLMs can list/load skill files.
Skill files live in _ex6/skills/*.md.
Uses yaml frontmatter for name/description. Filename = skill id fallback.

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


_skills_dir = Path(__file__).parent / "skills"



def _parse_frontmatter(text):
    """Parse yaml frontmatter from --- delimited block. Returns (metadata_dict, body)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            fm = yaml.safe_load("\n".join(lines[1:i]))
            body = "\n".join(lines[i+1:]).lstrip("\n")
            return fm or {}, body
    return {}, text


def _list_skills():
    if not _skills_dir.exists(): return []
    out = []
    for p in sorted(_skills_dir.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        fm, _ = _parse_frontmatter(text)
        sid = fm.get("name", p.stem)
        desc = fm.get("description", "")
        out.append((sid, desc))
    return out


def load_skill(ctx: ex6.Context, skill_id: str) -> str:
    if not skill_id:
        lines = [f"{sid}: {desc}" for sid, desc in _list_skills()]
        return "\n".join(lines) if lines else "(no skills are available!)"
    path = _skills_dir / f"{skill_id}.md"
    if not path.exists():
        avail = ", ".join(sid for sid, _ in _list_skills())
        raise ValueError(f"Unknown skill '{skill_id}'. Available: {avail}")
    text = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)
    name = fm.get("name", skill_id)
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


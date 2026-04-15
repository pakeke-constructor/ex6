"""
skills: Progressive disclosure. LLMs can list/load skill files.
Skill files live in _ex6/skills/*.md.
Filename = skill id. First line = short description. Rest = body.
"""
import ex6
from pathlib import Path


_skills_dir = Path(__file__).parent / "skills"


def _list_skills():
    if not _skills_dir.exists(): return []
    out = []
    for p in sorted(_skills_dir.glob("*.md")):
        first = p.read_text(encoding="utf-8").split("\n", 1)[0].strip()
        out.append((p.stem, first))
    return out


def load_skill(ctx: ex6.Context, skill_id: str) -> str:
    if not skill_id:
        lines = [f"{sid}: {desc}" for sid, desc in _list_skills()]
        return "\n".join(lines) if lines else "(no skills found)"
    path = _skills_dir / f"{skill_id}.md"
    if not path.exists():
        avail = ", ".join(sid for sid, _ in _list_skills())
        return f"Unknown skill '{skill_id}'. Available: {avail}"
    body = path.read_text(encoding="utf-8")
    ctx.messages.append(ex6.Message(role="user", content=f"[skill: {skill_id}]\n{body}"))
    return f"Loaded skill '{skill_id}'."



SKILL_DOC = "Load a skill's content into the conversation.\n"
_all_skills = _list_skills()
if not _all_skills:
    SKILL_DOC += "No skills exist."
else:
    SKILL_DOC += "Available skills:\n"
    for _sid, _desc in _all_skills:
        SKILL_DOC += f"  {_sid}: {_desc}\n"


load_skill.__doc__ = SKILL_DOC


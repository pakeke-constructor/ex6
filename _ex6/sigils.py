import re
import ex6
from ex6 import handler


SSOT = """\
The user wants to remind you to consider the statefulness of the relevant system/code.
(Bad state is one of the most common causes of bugs, and we want to avoid it.)
Some guidelines, in order:
- If it's possible to remove the state entirely via smarter code: REMOVE IT.
- Otherwise, if the state can't be removed, try make it a single-source-of-truth (SSOT).
- Otherwise, if state must be duplicated, then make sure the state is either short-lived or recomputed frequently.
- Lastly, if the bad state can't be short-lived or recomputed, think about a way to invalidate it, or make the consumers aware of the duplication.
Don't forget the bigger picture.
If the system/code is clean and minimal; that's fine, no changes needed.

(Rule of thumb: If a goblin comes in and changes a bunch of memory/state randomly;
the program should just change behaviour; nothing should ever become broken.)"""

GIT = """\
Check the git working tree before editing,
and check the resulting diff before finishing."""

FAST = """\
Move quickly via the smallest direct path.
Avoid broad exploration and optional work, but still verify."""

SMP = """\
Step back and look for a simpler solution.
If the work is growing or feels hacky, revisit the problem and constraints.
If it's already minimal, avoid churn."""

READ = """\
Prioritize readable code: clear control flow, shallow indentation, well-named variables."""

SHORT = """\
Make the code shorter: remove unnecessary code and comments, use concise forms,
but keep it readable."""

QUERY = """\
Do not write or edit any code; only answer the question or plan with the user.
Be concise and direct."""


SIGILS = {
    "ssot": SSOT,
    "git": GIT,
    "fast": FAST,
    "smp": SMP,
    "read": READ,
    "short": SHORT,
    "query": QUERY,
}
_SIGIL_RE = re.compile(rf";[ \t]*({'|'.join(SIGILS)})(?![\w-])", re.IGNORECASE)


@handler
def transform_user_prompt(text: str, ctx: ex6.Context) -> str:
    return _SIGIL_RE.sub(lambda m: f"\n<user_reminder>\n{SIGILS[m.group(1).lower()]}\n</user_reminder>\n", text)

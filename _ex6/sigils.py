import re
import ex6
from _ex6.models import M


SIGIL_MODEL = M.GEMINI31_FLASH_LITE.id
_SIGIL_RE = re.compile(r";([A-Za-z][\w-]*)")

SYSTEM_PROMPT = """\
You expand sigils in coding-agent prompts.
Return only short operational instructions implied by sigils. Do not answer or rewrite task.
Infer unknown sigils from surrounding prompt. Preserve user intent. Be concise.
"""


def transform_user_prompt(ctx: ex6.Context, text: str) -> str:
    sigils = list(dict.fromkeys(match.group(0) for match in _SIGIL_RE.finditer(text)))
    if not sigils:
        return text

    sub = ex6.Context("__sigils__", model=SIGIL_MODEL, reasoning="none", messages=[
        ex6.Message(role="system", content=SYSTEM_PROMPT),
        ex6.Message(role="user", content=f"Prompt:\n{text}\n\nSigils: {', '.join(sigils)}"),
    ])
    try:
        parts = []
        for item in ex6.invoke_llm(sub):
            if isinstance(item, ex6.ResponseChunk) and item.type == "text":
                parts.append(item.content)
        augmentation = "".join(parts).strip()
    finally:
        ex6.remove_context(sub)

    if not augmentation:
        return text
    return f"{text}\n\nSigil augmentation:\n{augmentation}"

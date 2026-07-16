
import ex6
import time
from _ex6.models import M


WEBSEARCH_SYSTEM_PROMPT = ex6.Message(role="system", content="""\
You are a focused web research agent with native web search enabled.
Answer the given question using up-to-date information from the web.

<output>
- Concise — facts only, no padding. Conciseness >> grammatical correctness.
- Plain text only, no markdown.
- Include specific details: code examples, parameter names, exact values.
- Cite source URLs inline for any non-trivial claim.
- If the web gives no clear answer, say so. Do NOT make things up.
</output>
""")


def websearch_agent(ctx: ex6.Context, question: str) -> str:
    """Spawn a websearch subagent to research a question. Returns a concise answer.
    Uses OpenRouter native web search. Use when you need up-to-date info from the web."""
    sub_name = f"websearch_{time.time_ns()}"
    sub = ex6.Context(sub_name, model=M.GEMINI_LATEST.id + ":online",
                      messages=[WEBSEARCH_SYSTEM_PROMPT], reasoning="none")
    sub.parent = ctx.name
    try:
        sub.invoke(question)
        while sub.llm_is_running:
            time.sleep(0.05)
        if sub.llm_result and sub.llm_result.error:
            raise RuntimeError(f"websearch_agent failed: {sub.llm_result.error}")
        messages = sub.get_messages()
        return messages[-1].content if messages else "No answer."
    finally:
        ex6.remove_context(sub)

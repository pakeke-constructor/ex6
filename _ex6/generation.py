import hashlib
from pathlib import Path

import ex6
from _ex6.models import M
from _ex6.provider_openai import invoke_llm


_INSTRUCTIONS = """
Write a concise system prompt for a specialist agent.

Extra info is user-authored: give it high importance. Let it shape the strategy
and philosophy, preserve its constraints, and prioritize it over generic advice.
First explore the local project with read-only tools: check documentation, layout,
and code relevant to the purpose and extra info. Keep exploration focused.
Use verified project facts to ground the prompt; do not invent tools or permissions.

Output only markdown with these four sections. Examples show input -> output;
adapt to the actual input rather than copying their requirements.
Length should fit the task: add detail when useful, not to match an example's size.

<example-1>
purpose: solidity analysis agent
extra info: Focus on gas fees. Measure savings; preserve contract behavior.

output:
```
Role:
You analyze Solidity contracts for gas efficiency.
Agent strategy steps:
- Inspect hot paths, storage access, and loops; establish baseline gas costs.
- Propose simple optimizations and measure savings under representative workloads.
- Verify that changes preserve contract behavior and security assumptions.
Agent philosophy:
- Prefer measured savings over cleverness.
- Never trade correctness or security for gas.
- Consider readability and maintenance costs alongside execution costs.
Extra details:
Preserve contract behavior.
```
</example-1>

<example-2>
purpose: code review agent
extra info: Prioritize regressions and simpler solutions. Do not edit or merge.
agent should be encouraged to look at the bigger picture.

output:
```
Role:
You review code for regressions and unnecessary complexity.
Agent strategy steps:
- Read the diff and understand the overall bigger picture before judging implementation.
- Reason out loud about the solution, and how it affects the overarching goal and bigger picture.
- Inspect relevant callers and tests to check assumptions and identify regressions.
- Look for smaller changes or simpler solutions that meet the same requirements.
- Report actionable findings with file locations, consequences, and suggested fixes.
Agent philosophy:
- Favor actionable evidence over style preferences or speculative concerns.
- Review the actual change, not an imagined future architecture.
- Prefer a few consequential findings over a long list of minor observations.
- Distinguish merge-blocking issues from optional improvements.
Extra details:
Do not edit code or merge changes. Explain uncertainty when a finding depends
on assumptions you could not verify.
```
</example-2>

After exploring the project, write a draft.
On review, rewrite the complete prompt to fix omissions, contradictions, vague steps, and bloat.
If no substantive fixes remain, output READY instead.
Do not wrap responses in code fences.
Most importantly, consider the broader project goals when designing the prompt.
"""


def _run(ctx):
    while True:
        message = ctx._read_llm_stream(invoke_llm)
        if ctx.llm_result.error:
            raise RuntimeError(ctx.llm_result.error)
        if ex6.call_tools(ctx, ctx.llm_result):
            continue
        ctx.append_message(message)
        ctx.pending_message = None
        return message.content.strip()


def generate_prompt(agent_purpose: str, xtra_info: str = "", *, cache_file: str | None = None, max_passes: int = 4) -> str:
    if cache_file is None:
        key = hashlib.sha256(repr((M.GPT_6_SOL.id, _INSTRUCTIONS, agent_purpose, xtra_info)).encode()).hexdigest()
        path = Path("_ex6/generation_cache") / f"{key}.txt"
    else:
        path = Path(cache_file)
    if path.exists():
        return path.read_text(encoding="utf-8")

    from _ex6.tools import glob, search, read_file, read_headers, read_body

    system = ex6.Message("system", _INSTRUCTIONS, tools=[glob, search, read_file, read_headers, read_body])
    ctx = ex6.Context(
        ex6.App(),
        "prompt-generation",
        M.GPT_6_SOL.id,
        reasoning="high",
        cwd=str(Path.cwd()),
        messages=[
            system,
            ex6.Message("user", f"Purpose:\n{agent_purpose}\n\nExtra info:\n{xtra_info}"),
        ]
    )
    ctx.append_message(ex6.Message("user", f"Explore the local project at {ctx.cwd} first. "
                                   "Finish with concise relevant project notes, not a draft."))
    _run(ctx)
    ctx.append_message(ex6.Message("user", "Now write the system prompt using what you learned."))
    prompt = ""
    for _ in range(max_passes):
        response = _run(ctx)
        if response == "READY" and prompt:
            break
        prompt = response
        ctx.append_message(ex6.Message("user", "Review the prompt. Rewrite it, or output READY if satisfied."))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    return prompt

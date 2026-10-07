import ast
import hashlib
import inspect
from pathlib import Path
from pprint import pformat

import ex6
from _ex6.models import M
from _ex6.provider_openai import invoke_llm


_INSTRUCTIONS = """
Write a concise system prompt for a specialist agent.
Extra info is user-authored: give it high importance. Let it shape the strategy
and philosophy, preserve its constraints, and prioritize it over generic advice.
Do not invent repository facts, tools, or permissions. You have no codebase access.

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

First write a draft. On review, rewrite the complete prompt to fix omissions,
contradictions, vague steps, and bloat. If no substantive fixes remain, output
only READY instead. Do not wrap responses in code fences.
"""


def _cache(lines):
    for node in ast.parse("".join(lines)).body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == "_GENERATED_PROMPTS" for t in node.targets):
            return ast.literal_eval(node.value), node.lineno - 1, node.end_lineno
    return {}, len(lines), len(lines)


def generate_prompt(agent_purpose: str, xtra_info: str = "", *, max_passes: int = 4) -> str:
    if max_passes < 2:
        raise ValueError("Prompt generation needs a draft and at least one review")
    path = Path(inspect.currentframe().f_back.f_code.co_filename).resolve()
    key = hashlib.sha256(repr((M.GPT_6_SOL.id, _INSTRUCTIONS, agent_purpose, xtra_info)).encode()).hexdigest()
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    cached, _, _ = _cache(lines)
    if key in cached:
        return cached[key]

    ctx = ex6.App().create_context("prompt-generation", M.GPT_6_SOL.id, reasoning="high", messages=[
        ex6.Message("system", _INSTRUCTIONS),
        ex6.Message("user", f"Purpose:\n{agent_purpose}\n\nExtra info:\n{xtra_info}"),
    ])
    prompt = ""
    for _ in range(max_passes):
        response = ""
        for chunk in invoke_llm(ctx):
            if isinstance(chunk, ex6.LLMResult):
                if chunk.error:
                    raise RuntimeError(chunk.error)
            elif chunk.type == "text":
                response += chunk.content
        response = response.strip()
        if response == "READY" and prompt:
            break
        prompt = response
        ctx.append_message(ex6.Message("assistant", prompt))
        ctx.append_message(ex6.Message("user", "Review the prompt. Rewrite it, or output READY if satisfied."))
    else:
        raise RuntimeError(f"Prompt generation did not converge after {max_passes} passes")

    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    cached, start, end = _cache(lines)
    cached[key] = prompt
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    entry = "_GENERATED_PROMPTS = " + pformat(cached, width=100) + "\n"
    if start == len(lines):
        entry = "\n\n" + entry
    lines[start:end] = [entry.replace("\n", newline)]
    path.write_bytes("".join(lines).encode("utf-8"))
    return prompt

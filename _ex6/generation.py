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
Preserve the supplied purpose and extra details. Do not invent repository facts,
infrastructure, tools, or permissions. You have no codebase access; instructions
to inspect it belong in the specialist's strategy. Prefer pragmatic, simple steps.

Output only a markdown blob with these sections, in this order:
Role:
You are ...

Agent strategy steps:
- ...

Agent philosophy:
- ...

Extra details:
...

Examples of the intended structure and specificity, not requirements to copy:

Example: code-review agent
Role:
You are a code reviewer. Find consequential problems and simpler solutions.

Agent strategy steps:
- Read the diff and understand the intended change.
- Inspect surrounding code and callers to verify assumptions.
- Check correctness, regressions, security boundaries, and test coverage.
- Look for a smaller change that achieves the same goal.
- Report actionable findings with locations, consequences, and suggested fixes.
- Separate merge-blocking issues from optional improvements.

Agent philosophy:
- Correctness and simplicity matter more than stylistic preferences.
- Review the actual change, not an imagined future architecture.
- Prefer a few well-supported findings over speculative concerns.

Extra details:
Do not modify code or merge changes unless explicitly requested.

Example: Solidity contract analysis agent
Role:
You analyze Solidity contracts for vulnerabilities and economic failure modes.

Agent strategy steps:
- Identify assets, privileged roles, dependencies, and trust assumptions.
- Trace asset flows and state transitions through external entry points.
- State intended invariants and check authorization and accounting against them.
- Examine reentrancy, upgrades, signature replay, rounding, and token behavior.
- Check oracle assumptions, liquidation logic, and transaction-ordering risks.
- Validate realistic attack sequences with focused tests or proofs of concept.
- Report affected code, prerequisites, exploit path, impact, and mitigation.

Agent philosophy:
- Follow assets and state transitions, not just vulnerability checklists.
- Separate permissionless exploits from explicitly trusted administrator powers.
- Treat economic assumptions as part of the security model.
- Explain attacker benefit or concrete harm before calling an issue exploitable.
- State uncertainty and coverage; no findings does not prove safety.

Extra details:
Validate locally or on forks. Do not submit transactions to live networks.

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

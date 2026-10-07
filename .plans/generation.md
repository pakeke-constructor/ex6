# Specialist prompt generation

Motivation: make singular-purpose specialist agents easy to define, with minimal code and prompts visible/editable in the local plugin that requested them.

- `_ex6/generation.py`: `generate_prompt(agent_purpose, xtra_info="", *, max_passes=4)`.
- Use GPT-6 SOL through the existing Codex subscription provider.
- On cache misses, explore the current local project first with `glob`, `search`, `read_file`, `read_headers`, and `read_body`. Keep project notes and tool results in the generation context; disable tools before drafting/reviewing. Ground specialist prompts in verified project facts and broader goals.
- Output a markdown blob directly: `Role:`, `Agent strategy steps:`, `Agent philosophy:`, `Extra details:`. No JSON schema, validation, or rendering layer.
- Draft, then review/rewrite. A review returning `READY` accepts the last draft; fail without caching if the pass limit is reached.
- Cache in a literal `_GENERATED_PROMPTS` dictionary in the calling plugin. Read from source so module-level calls work before the dictionary executes. Changed purpose, extra info, model, or instructions regenerate.
- Cache hits skip exploration; delete an entry to refresh project knowledge after codebase changes.
- Keep source/newlines intact. Verify with a temporary offline smoke test, then delete it. No permanent test classes.

Relevant files: `_ex6/generation.py`, `_ex6/models.py`, `_ex6/provider_openai.py`.

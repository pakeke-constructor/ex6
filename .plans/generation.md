# Specialist prompt generation

Motivation: make singular-purpose specialist agents easy to define, with minimal code and prompts visible/editable in the local plugin that requested them.

- `_ex6/generation.py`: `generate_prompt(agent_purpose, xtra_info="", *, max_passes=4)`.
- Use GPT-6 SOL through the existing Codex subscription provider.
- Output a markdown blob directly: `Role:`, `Agent strategy steps:`, `Agent philosophy:`, `Extra details:`. No JSON schema, validation, or rendering layer.
- Draft, then review/rewrite. A review returning `READY` accepts the last draft; fail without caching if the pass limit is reached.
- Cache in a literal `_GENERATED_PROMPTS` dictionary in the calling plugin. Read from source so module-level calls work before the dictionary executes. Changed purpose, extra info, model, or instructions regenerate.
- Keep source/newlines intact. Verify with a temporary offline smoke test, then delete it. No permanent test classes.

Relevant files: `_ex6/generation.py`, `_ex6/models.py`, `_ex6/provider_openai.py`.


# TASK: Fix OpenRouter Param Passthrough

## Context
Dependency T-002 confirmed that the `anthropic` header is being stripped. 
We need to wrap this in `extraParams`.

<plan>
1. Identify the request builder in `client.py`.
2. Move `max_tokens` and `temperature` into the `extraParams` dict.
3. Test with a dry run.
</plan>


<log>
[2026-03-17T22:12:00Z] Started investigation.
[2026-03-17T22:15:00Z] [FINDING]: The SDK version 1.2.0 changed the signature.
[2026-03-17T22:19:00Z] Commit b45982d fixes the bug.
</log>


<meta>
status: in-progress
created_at: 2026-03-17T22:10:00Z
creator: Agent-Alpha
relates_to: [T-001]
dependencies: [T-002]
confidence: 0.7
</meta>




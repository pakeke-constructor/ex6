# invokelogs

## Motivation (user's words)
"I wasn't aware that I was collecting so many invoke calls."
Use this data to try to extract insights. "Hopefully we can use these insights to make our agents more useful."

## Current state
- `%APPDATA%\ex6\logs\invoke-*.txt`: ~62,600 files, ~4.2 GB (2026-02-01 to 2026-10-28).
- Written by `_log_invoke` in `_ex6/provider.py`, called from provider.py, provider_openai.py, provider_anthropic.py.
- Problem: "most of the stuff is badly formatted, it doesn't show output upon completion, it only shows the initial system-prompt a lot of the time."
  - Observation: several call sites pass `[]` as messages (provider_anthropic.py:397, :553; provider_openai.py:167), so many files have a header but no content.

## Tasks
1. Quick idempotent cull of all the log files without any content in them.
2. Use the remaining data to extract insights to make our agents more useful.
3. Going forwards, keep logging invocations, but log on completion, not on invoke.
   - "This will be much heavier in terms of space, so I want to maybe optimize this a little bit; since context caching is a thing, maybe we can be smart about having references to blobs of text or something?"

## Status
Plan only. No solutions chosen yet. Discuss with the user before designing or implementing anything.

## Files
- `_ex6/provider.py` (`_log_invoke`, `_log_safe`)
- `_ex6/provider_openai.py`
- `_ex6/provider_anthropic.py`
- `ex6.get_folder()` in `ex6.py`

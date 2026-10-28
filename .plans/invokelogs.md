# invokelogs

## Motivation (user's words)
"I wasn't aware that I was collecting so many invoke calls."
Use this data to try to extract insights. "Hopefully we can use these insights to make our agents more useful."

## Current state
- `%APPDATA%\ex6\logs\invoke-*.txt`: ~62,600 files, ~4.2 GB (2026-02-01 to 2026-10-28).
- Written by `_log_invoke` in `_ex6/provider.py`, called from provider.py, provider_openai.py, provider_anthropic.py.
- Problem: "most of the stuff is badly formatted, it doesn't show output upon completion, it only shows the initial system-prompt a lot of the time."

## Tasks
1. Clear the folder and start fresh. The old logs can safely be deleted (`%APPDATA%\ex6\logs\invoke-*.txt`). This replaces the "cull empty files" idea.
2. FIX (must be fixed): several call sites pass `[]` as messages, so their logs have a header but no conversation:
   - `_ex6/provider_anthropic.py:397`
   - `_ex6/provider_anthropic.py:553`
   - `_ex6/provider_openai.py:167`
   Every call site must log the actual messages sent.
3. Going forwards, keep logging invocations, but log on completion, not on invoke (include the model's output).
   - "This will be much heavier in terms of space, so I want to maybe optimize this a little bit; since context caching is a thing, maybe we can be smart about having references to blobs of text or something?"
4. Once fresh logs accumulate, use them to extract insights to make our agents more useful.

## Status
Plan only. No design chosen for task 3 (storage format / blob references). Discuss with the user before designing it.

## Files
- `_ex6/provider.py` (`_log_invoke`, `_log_safe`)
- `_ex6/provider_openai.py`
- `_ex6/provider_anthropic.py`
- `ex6.get_folder()` in `ex6.py`

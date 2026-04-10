# Task: Fix message spacing in render_work_mode

## Problem
There is unwanted vertical spacing between tool-call groups in the conversation view.

## How messages work
- Messages have roles: system, user, assistant, tool
- Tool-call flow: assistant (with tool_calls, often empty content "") -> tool result -> assistant (with tool_calls) -> tool result -> ... -> assistant (final answer)
- In code_mode, `ctx._tool_renderers` maps tool_call_id -> a callable that renders tool status lines
- During a live run, _tool_renderers is populated: tool messages are skipped, callable renderers are appended to assistant message lines
- After reset/reload, _tool_renderers is cleared: tool messages are no longer skipped, callables are not appended

## What we want
- Tool-call groups (assistant+tools -> tool results -> assistant+tools -> ...) should be visually bunched together with ZERO spacing
- User and assistant (final answer) messages should have 1 line of spacing around them
- Tool result messages should be invisible to the user (they are for the LLM only)
- Empty assistant messages (content="" with only tool_calls) should take up zero visual space

## Key files
- ex6.py: render_work_mode (~line 1265), _tool_renderers, set_tool_renderer
- _ex6/code_mode.py: code_render callable (line 247), set_tool_renderer call (line 259)
- _ex6/provider.py: _cc_strip_tool_blocks output_renderer

## Desired end state
A conversation with multiple tool-call rounds looks like:

user message

assistant text (if any)
[v] tool_name(args)
[v] tool_name(args)
[v] tool_name(args)
assistant text (if any)
[v] tool_name(args)
[v] tool_name(args)

assistant final answer

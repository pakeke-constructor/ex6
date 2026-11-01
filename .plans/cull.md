# cull: remove tech debt left over from agents-mode / work-mode

## Motivation
User: "Removal of tech debt. with this refactor we can probably cull a lot of old shit."
`unify` merged agents-mode and work-mode into one screen (list + transcript + input, NORMAL/INSERT focus, `w` zoom). Lots of code, names and docs still assume two modes. ex6 is meant to be a thin, hyper-minimal layer: dead paths and stale names cost every future agent reading ex6.py tokens and confusion. Goal: net negative lines, no behavior change (except deleting things nobody needs).

## Candidates (verify each before deleting; grep `_ex6/`, `_plugin_examples/`, user `get_folder()/_ex6`)
Naming (two-mode vocabulary):
- `render_work_mode` -> eg `render_transcript`; `render_work_mode_input` -> `render_input`; `render_work_mode_footer` handler (provider_openai/anthropic use it); `render_workmodefooter_and_commands` (also has a stale rambling "todo" docstring); `render_agents_mode_context_name`, `render_agents_left/right` (right is now just the header). Overrides/handlers are referenced by name in plugins + loading_cube example: rename everywhere at once.
- `sel_left_w`, `# --- AGENTS UI ---` section split, comments saying "agents mode"/"work mode" in ex6.py, `_ex6/*.py`, `AGENTS.md`.

Possibly dead / duplicated:
- `TUI.input_box` field (set every frame; does anything read it?).
- `TUI.prev_mode` / scroll-mode restore: still needed only for `/dbg`; check whether scroll mode can be simpler.
- `/yolo` command in ex6.py core vs `z` key in vim.py (and core shouldn't define plugin-ish commands at all?). `/dbg` likewise.
- `Message.overview` comment says "agents panel": only z_render_system_prompts uses it now.
- `make_input` overridable: does anything override it?
- `show_cot`: anything toggles it?
- z_render_system_prompts' `expanded` state: never toggled (no click handling exists); the expanded branch is dead code.
- `_one_line` now has a single caller area (status line).
- `render_tool_line` / `_tool_row` / `ToolCall`: check still minimal after unify.

Repo clutter (ask user before deleting):
- `scroll_proto.py`, `rough_planning/`, `_notes/`, `fun_projects.md`, `claude-code-system-prompts/`.
- Any `.plans/*.md` still written against two modes ("work mode", "selection mode"): update or prune.

## Approach
- One commit per group (renames, dead code, docs, clutter) so it's easy to review/revert.
- After each: run a fake-terminal smoke frame loop (see how `unify` was checked: blessed Terminal on StringIO, call `_tui_loop` a few times, print `buf._prev_chars`), then delete the script.
- Update `AGENTS.md` architecture/core-plugins sections for any renamed overridable/handler.

## Files
- `ex6.py`, `AGENTS.md`
- `_ex6/vim.py`, `_ex6/commands.py`, `_ex6/z_render_system_prompts.py`, `_ex6/provider_openai.py`, `_ex6/provider_anthropic.py`, `_ex6/profiler.py`, `_ex6/_plugin_examples/loading_cube.py`
- `.plans/*.md`

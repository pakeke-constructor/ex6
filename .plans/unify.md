# unify: work-mode content always visible in agents-mode; drop tools side-panel

## Motivation
User: "make it so the work-mode stuff is visible in agents-mode, at all times, so users dont need to leave agents-mode to do things." Also: "could the tools side-panel be removed?"
Today agents-mode right panel is a lossy summary (`_overview_rows`), so you jump to work mode to actually read output, scroll, or answer. Two modes = two layouts, two Esc/arrow paths, two places a human panel can show. One screen is simpler and more vim-like (list = normal mode, input box = insert mode).

## Layout
```
 1 explore (12k)   | coder  anthropic/opus · provider_anthropic        $1.20 / $20
 0 coder (40k) ●   | ████░░░░ 40k / 200k 20%
 1 sound (3k)      |----------------------------------------------------------
                   | <render_work_mode transcript, own _scroll_up per ctx>
                   |
                   |----------------------------------------------------------
                   | > input box (always drawn; dim when unfocused; spinner/queued as today)
                   |   /command autocomplete footer
 NORMAL coder · low  yolo OFF   <echo>                                    <keys>
```
- Right panel = header (name/model/via/budget + ctx bar, from `render_agents_right`) + `render_work_mode(tui, buf, inpt, right_transcript_r)` + `render_work_mode_input`. render_work_mode already takes a region; drop its own header row (duplicate).
- Left panel unchanged (auto width via `sel_left_w`).
- Remove `_overview_rows` / `_group_agent_turns` / `_agent_turn_rows` / `_tokstr` (transcript supersedes them). Maybe keep the 1-line "last activity" idea for the left list later, not now.

## Focus instead of modes
- `tui.mode`: "agents"/"work" -> focus: list (keys active) vs input (typing). Basically today's `sel_input`, renamed. "scroll" mode stays.
- space/a/h focus input; Esc returns to list. Enter on empty input: no-op (or "zoom", below).
- Arrows: list focus -> j/k nav (as now); input focus -> scroll transcript. PgUp/PgDn scroll in both. Add `<C-u>`/`<C-d>` keys in vim.py for list focus scrolling.
- Status line label: NORMAL / INSERT (or LIST / INPUT).
- `last_seen`: set when ctx is current (you're looking at it). `●` = finished while not selected.

## Human panels (ctx.ui_stack)
- Draw inside transcript region instead of full-screen overlay, whenever current has one -> question visible immediately on selecting the agent.
- Only pass real `inpt` when input-focused; otherwise an empty InputPass (read-only preview). Needs check that draw fns tolerate a small region.
- `ui_panel_stack` (app-level modals, eg /help) stays full-screen overlay.

## Zoom (optional, small)
- `w` key (wide; free in vim.py) toggles `tui.zoom`. On: `left_w = 0`, header collapses to one row (old work-mode header), status line shows `ZOOM`.
- Why: terminal mouse-selection grabs whole rows, so the left list pollutes copied text; also long code lines.
- Sticky: survives j/k (cycle agents zoomed), focus changes, Esc. Only `w` toggles. Enter-on-empty stays no-op.
- ~10 lines. Replaces work mode for those who want it.

## Tools side-panel
Remove `_render_tools_column`. Reasons: eats ~1/3 of width at narrow terms, static info, doesn't belong next to a transcript.
But "every token explicit" -> tools must still be discoverable:
- Header row 2, after ctx bar: `· 14 tools (2 native)` muted.
- Full list: z_render_system_prompts' expandable system message lists tool names + schema token cost per tool (tools are part of the system message). User confirmed panel removal.

## Estimate
~150-200 lines touched, net about -50. Deletes: tools column + overview helpers + work-mode header (~-85). Adds: tool list in expander (~+20), vim scroll/zoom keys (~+20), `_tui_loop` layout rewrite (~40 changed, roughly net 0), human panel in-region (~+5).

## Risks / open questions
- Perf: transcript renders every frame for current ctx only; same cost as work mode today. Output renderers cache by text, not width: fine.
- Narrow terminals: left ~20 cols + transcript. Code blocks wrap more; zoom covers it.
- Plugins overriding `render_work_mode` (loading_cube example) keep working since signature unchanged. `render_agents_right` override semantics change (now only header).
- Keybinding `?` help text and CLAUDE/arch docs (`<ui-ux>` section, vim.py docstrings) need updating.
- Overlaps `.plans/interact.md` and `.plans/remap.md` (Esc handling consolidates to one place here: good for remap).

## Files
- `ex6.py`: `_tui_loop` (layout, Esc/arrow/Enter handling), `TUI.mode`/`sel_input`/`keys_active`, `render_agents_right`, `_render_tools_column`, `_overview_rows` & helpers, `render_work_mode` (header row), `render_work_mode_input`, `render_status_line`
- `_ex6/vim.py`: `prompt` docstring (Enter=work mode), `_open_input`, new scroll/zoom keys, `help_lines`
- `_ex6/z_render_system_prompts.py`: tool list in expander
- `_ex6/profiler.py`: prints `tui.mode`
- `_ex6/_plugin_examples/loading_cube.py`: check still fine

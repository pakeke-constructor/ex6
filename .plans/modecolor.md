# modecolor: make NORMAL vs INSERT obvious at a glance

## Motivation
User: "The UX for knowing whether user is in NORMAL or INSERT mode is... really bad. Maybe all the lines on screen could be made blue when in INSERT mode or smthn? (AGENTS is then green, INSERT is blue?)"
After `unify`, agents-mode and work-mode are one screen; the only mode signal is the status-line label + a dimmed input box. Easy to type `dd` thinking you're in the list, or type a prompt into the list. The mode must be readable from peripheral vision, not by reading a label.

## User's idea
- All divider/border lines on screen take the mode color: green in NORMAL (list focused; user also called it "AGENTS"), blue in INSERT.
- Status-line label swaps to match. Today it's the opposite (NORMAL = `th.accent` blue, INSERT = `th.success` green): flip it.
- Maybe rename the NORMAL label to AGENTS (user's word). Decide.

## Things to check / decide
- Which lines: left panel vline (`render_agents_left`), header divider + `├` joins (`render_agents_right`), input divider in `_tui_loop` (currently `th.invoking` when running: running signal must survive, eg spinner only, or only the input divider keeps invoking color), selected-row bg in the list?
- Theme: add `th.mode_normal` / `th.mode_insert` (themes.py palettes too) vs reuse success/accent.
- One place to pick the color (eg `tui.mode_color()` or computed once in `_tui_loop` and passed/stored), not scattered `if tui.mode == ...` checks.
- Zoom has no left vline: dividers alone must carry it.
- Other cues worth trying cheaply: hide the input cursor `█` when unfocused; brighter list selection in NORMAL vs dim in INSERT.

## Files
- `ex6.py`: `_tui_loop` (input divider), `render_agents_left`, `render_agents_right`, `render_status_line`, `render_work_mode_input` (dim/cursor), `Theme`
- `_ex6/themes.py`: palettes if new theme colors

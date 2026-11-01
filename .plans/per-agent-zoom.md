# Per-agent transcript zoom

## Motivation
Keep navigation consistent: `j`/`k` and arrow keys always select agents in NORMAL mode, whether the list is visible or not. Zoom only controls the selected agent's transcript layout, and each agent remembers its own layout when switching away and back.

## Behavior
- `w` toggles zoom for the current agent only.
- `j`/`k` and Up/Down continue selecting agents while zoomed.
- `Ctrl-U`/`Ctrl-D` and PgUp/PgDown continue scrolling transcript history.
- Switching agents restores that agent's zoom state.
- New and forked agents start unzoomed; clearing an agent preserves its zoom state.

## Changes
- `ex6.py`
  - Move zoom state from `TUI.zoom` to a private boolean on `Context`.
  - Reset zoom in `Context._clone` so forks start unzoomed.
  - Read current context's zoom state in layout, header, and status-line rendering.
- `_ex6/vim.py`
  - Make `w` toggle `tui.current`'s zoom state.
  - Leave agent navigation and transcript scrolling bindings unchanged.

## Verification
- Zoom agent A, switch to unzoomed agent B with both `j`/`k` and arrows, then return to A and confirm zoom is restored.
- Confirm `Ctrl-U`/`Ctrl-D` and PgUp/PgDown scroll history in both layouts.
- Confirm new/forked agents are unzoomed and `/clr` does not change zoom.

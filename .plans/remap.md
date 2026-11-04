# remap: NORMAL-mode key remaps (nnoremap)
Status: parked

## Motivation
User: "regular vim allows for `nnoremap kj esc`. we want that too." INSERT remaps are done (`app.remap(lhs, rhs, mode="insert")`, resolved in `InputBox.handle`; `jk`/`kj` -> Esc set in `_ex6/commands.py`). NORMAL and `ui_panel_stack` modals (eg `?` help) have no remaps yet.

## Remaining
- `mode="normal"`: resolve in `TUI.feed_keys` alongside `app.keys` (same no-timeout prefix rules as `app.bind`; a remap and a binding sharing a prefix must raise). rhs keys re-fed without remapping.
- Modals: decide whether NORMAL remaps apply while `ui_panel_stack` is open (its Esc pop happens at the top of `_tui_loop`).

## Files
- `ex6.py`: `App.remap`, `token_key`, `TUI._resolve_keys`/`feed_keys`, `_tui_loop` Esc handling

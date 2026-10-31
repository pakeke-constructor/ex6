# remap: vim-style key remaps, eg `kj` -> Esc

## Motivation
User: "is it possible to do `@key("kj")` bind to escape?" "regular vim allows for `nnoremap kj esc`. we want that too." Wants it done "the proper way", not as a hack.

## Problem
There's no way to map one key sequence to another key, like Esc. Today:
- `@ex6.key` binds a sequence to a Python fn, and only in selection mode. It can't produce a key, and it does nothing while you type in an input box (work mode, the space/`a` sel input, ask panels).
- With no timeouts, `app.bind` rejects any sequence that starts with another binding. `kj` starts with `k` (prev agent), so it's rejected.
- Esc is handled in several places in `_tui_loop` (ui_panel_stack pop, sel_input close, work -> selection), plus inside panels. So "send Esc" doesn't go to one place.

Open questions: which modes a remap applies in, how ambiguous prefixes resolve, and what happens to typed text that might be the start of a remap.

## Files
- `ex6.py`: `key`, `split_keys`, `key_token`, `App.bind`, `TUI._resolve_keys`/`feed_keys`, `InputPass`, `InputBox.draw`, `_tui_loop` Esc handling
- `_ex6/vim.py`

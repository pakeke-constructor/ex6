# remap: vim-style key remaps, eg `kj` -> Esc

## Motivation
User: "is it possible to do `@key("kj")` bind to escape?" "regular vim allows for `nnoremap kj esc`. we want that too." Wants it done "the proper way", not as a hack.
After `unify` there's one screen with NORMAL (list focused, keys active) and INSERT (typing in the selected agent's input box). `kj` -> Esc is mostly wanted in INSERT, to get back to the list without reaching for Esc.

## Problem
There's no way to map one key sequence to another key, like Esc. Today:
- `@ex6.key` binds a sequence to a Python fn, and only in NORMAL (`TUI.keys_active`). It can't produce a key, and it does nothing in INSERT, in human panels (`ctx.ui_stack`, answered in INSERT), or in `ui_panel_stack` modals (eg `?` help).
- With no timeouts, `app.bind` rejects any sequence that starts with another binding. `kj` starts with `k` (prev agent), so it's rejected.
- Esc is now handled in two places at the top of `_tui_loop` (pop `ui_panel_stack`, INSERT -> NORMAL), before keys are fed. A remap that emits Esc must land before those checks (eg rewrite `inpt._keys` up front), so it reaches the same place a real Esc does.

Open questions: which modes a remap applies in (vim: `nnoremap` vs `inoremap`; here NORMAL vs INSERT), how ambiguous prefixes resolve, and what happens to typed text that might be the start of a remap (INSERT `k` must show up as you type it, then get removed if `j` follows; vim uses a timeout, ex6 has none).

## Files
- `ex6.py`: `key`, `split_keys`, `key_token`, `App.bind`, `TUI._resolve_keys`/`feed_keys`/`keys_active`, `InputPass`, `InputBox.draw`, `_tui_loop` Esc handling
- `_ex6/vim.py`

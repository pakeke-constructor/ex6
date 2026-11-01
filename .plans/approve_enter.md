# approve_enter

## Motivation (user's words)
"when i press `h` to go to the nearest agent that needs me, i see `approve bash tool`. i read it, and then i press enter to approve. However, enter doesnt approve it, it puts me in INSERT Mode. then, i need to press enter AGAIN to approve it."
"make the panel ITSELF more natively aware of normal-mode. and then, it can seamlessly integrate, not work against it."
"we could reuse keybind/muscle memory from normal/insert mode". Add `i` alongside space/a.
Rejected: timestamp/Enter-forwarding hacks; loop juggling InputPass per mode; panels reading `ctx.app.tui.mode`.

Root cause: ctx.ui_stack panels only get keys in INSERT. `h` opens INSERT, so the first approval works; answering drops to NORMAL, the next approval shows up in NORMAL, and Enter hits the `KEY_ENTER -> insert` fallback.

## Design: panels declare NORMAL-mode keys (keymap layer)
Two clean channels, no mode checks inside panels:
- INSERT: panel `draw(buf, inpt, r)` gets keys, exactly as today (typing, arrows, Enter). Unchanged.
- NORMAL: panel contributes key bindings (`{"<CR>": value}`) to the keymap while it's top of the selected agent's ui_stack. Pressing one answers the panel with `value`. Resolved by the normal keymap machinery (feed_keys, counts, which-key), so the panel is "native" in NORMAL.

Changes:
1. `Context.ask_human(draw, keys=None)` and `request_approval(ctx, tool, args, draw, keys=None)`: keys = {seq: value}. Store on the panel closure as `panel.keys = {split_keys(seq): fn(tui, n) -> answer(value)}` (same style as `fn._ex6_keys`).
2. `TUI.keymap()`: `{**app.keys, **top panel keys of current}`. `_resolve_keys` + `key_continuations` use it. Panel keys shadow core keys.
3. `_approve` (tools.py) passes `keys={"<CR>": True}`. Hint: `ENTER approve | i: type reason + ENTER to deny`.
4. `key_prompt` binds `" ", "a", "i"`.
5. Loop routing, `h`, Enter fallback, mode reset: unchanged.
6. AGENTS.md UI/UX lines 166-171: panel keys, `i`.

Not given NORMAL keys: ask_user, ask_user_question, escalate (answer needs reading/choosing; Enter in NORMAL still focuses them). Digits are counts in NORMAL, so 1-9 quick-pick stays INSERT-only.

Edge cases:
- Stacked approvals: only top panel's keys active.
- j/k switching: layer follows `tui.current`.
- `s`/stop: removes panel as today.
- Risk: Enter in NORMAL meant "focus input"; on an approval panel it now approves. Accepted (that's the ask).
- Pre-existing, out of scope: approval arriving while typing a prompt in INSERT steals keystrokes into the reason box.

~25 lines total.

## Files
- ex6.py: `Context.ask_human` (~1433), `request_approval` (~2193), `TUI._resolve_keys`/`key_continuations` (~2437), `key_prompt` (~2597), `_tui_loop` (~2792, for reference only)
- _ex6/tools.py: `_approval_panel` (~1443), `_approve` (~1469)
- AGENTS.md (CLAUDE.md imports it)

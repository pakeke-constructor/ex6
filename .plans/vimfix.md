# vimfix: repair the vibe-coded selection-mode keys, don't tear them down

## Motivation
The uncommitted interact work (keys.py, prompts.py, status line, `;name`, ask 1-9 tap) points the right way but was vibe-coded. User: "i want to build off it and repair it; not tear it down." "LESS CODE IS BETTER, SIMPLER CODE IS BETTER."
Design it serves (from `.plans/interact.md` human note): work-mode = typing prompts, selection-mode = keybinds, no other modes. In selection-mode you can still type and dispatch prompts without opening work-mode.

## User's asks
1. Rename `_ex6/keys.py` -> `_ex6/vim.py` (more accurate). Update imports (commands.py `/help`), AGENTS.md.
2. "We need a more formal way to say submit". Hacky code in `ask_user_question`:
   `for i in range(9): if inpt.consume(str(i+1)): result[0] = ...; if draw in ctx.ui_stack: ctx.ui_stack.remove(draw); return`
3. New key `a` in sel-mode: opens the input box of the agent. (Open: how is this different from `space`? Guess: same action, `a` is easier to hit and reads as vim "append". Ask the user before building anything more than an alias.)
4. `expand_abbrev` is badly named. Should be `transform_user_prompt`, which already exists. "why do we have both?"
5. "remove ya and yl, replace with `yc` = yank code, and `yr` = yank response."
6. "is it possible to do `@key("kj")` bind to escape?" (moved to `.plans/remap.md`)

## 2. Formal "submit" for human panels (ctx.ui_stack)
Problem: same boilerplate in 4 tools in `_ex6/tools.py` (~1195, ~1293, ~1353, ~1503): `push_ui(draw)`, `while draw in ctx.ui_stack: if stop_early: remove; sleep`, and `result[0] = x; if draw in ctx.ui_stack: ctx.ui_stack.remove(draw)` scattered in every exit path. ask_user_question alone has 3 copies.
Fix: one core primitive, `ctx.ask_human(draw) -> value`. The tool thread blocks until the panel submits or `stop_early` is set (then it returns None). `draw(buf, inpt, r)` returns the answer to submit, or None to keep showing. Core `_tui_loop` pops the panel and hands the value back. `push_ui` gets deleted, and the tools drop their `result = [None]` lists, `while` loops and `remove` calls.
- ask_user_question number tap then becomes `if inpt.consume(str(i+1)): return options[i]`.
- The `make_input` on_submit callback inside a panel just sets a local and the draw fn returns it (or InputBox.draw returns the submitted InputValue, see below).
- Bug to fix along the way: typing `1` as the first char of a custom answer picks option 1. OK for now, but write it down.

The same "who submitted?" hack exists in `_tui_loop` selection branch:
`submitting = inpt.peek('KEY_ENTER') and input_box.text; input_box(...); if submitting and not input_box.text: tui.sel_input = False`
Fix: InputBox.draw returns the InputValue it submitted (still calls on_submit). Caller does `if input_box(...): tui.sel_input = False`. One mechanism for panels and the sel input.

## 3. `a` key
Bind in vim.py next to `space`: `@key(" ", "a")` on `prompt`. That's 0 new lines if it's an alias.

## 4. One prompt hook: `transform_user_prompt`
Now: `transform_user_prompt` (pipeline, hidden, runs in `Context.invoke` thread on submit) has ZERO handlers since sigils.py was deleted. `expand_abbrev` (call, live in InputBox on non-word char + on Enter) does `;name`. Two hooks plus a regex in core InputBox (`_expand_abbrev`, `tail` logic, `app=` param on InputBox).
Fix: keep only `transform_user_prompt(text, ctx)` as the pipeline, and move where it runs: InputBox Enter, not invoke. If the pipeline changes the text, show the result and don't send. The next Enter sends (the "show expansion first" behaviour that's already there). That keeps "no hidden context": the user always sees what gets sent.
- prompts.py owns the `;name` / `;;name` regex inside its handler. Core InputBox loses `_expand_abbrev`, the `re` usage, and `app` (it needs ctx or a transform fn: `Context.get_input_box` passes it).
- Remove the pipeline call from `Context.invoke`: programmatic `ctx.invoke(text)` (subagents, horizon) sends raw text.
- Lost: live expansion on space/punctuation. Fine: Enter-to-preview covers it with less code. If the user misses it, add it back as a `transform_user_prompt` call on non-word chars.
- Delete the `expand_abbrev` handleable and its AGENTS.md mentions.

## Other issues found (simplifications)
- `/help` and `?` duplicate each other. commands.py lazily imports `help_lines` from keys.py, and keys.py imports `_text_panel` from commands.py (circular). Move `/help` into vim.py, delete it from commands.py.
- `KeyArgs.has_count` is unused. Delete it (KeyArgs is then just `count`; maybe pass `count: int` directly and drop the dataclass).
- Core vs vim.py: core keeps the key machinery (`@ex6.key`, `app.bind`, parser, `echo`, status line + which-key), like `@ex6.command`. vim.py keeps all the bindings (taste, deletable).
- Nav: core handles leftover keys. User: "`enter` enters into workmode IFF there is no text in input box (ie nothing consumed it)". In selection mode, core consumes arrows and Enter only after plugins, the sel input box and panels have drawn (end of the selection branch), so bare ex6 stays navigable. Delete vim.py's `<CR>` binding. Free gesture: `space`, then Enter on an empty box = open in work mode.
  - InputBox: `inpt.consume('KEY_ENTER') and self.text` swallows Enter on an empty box. Flip it to `self.text and inpt.consume(...)`.
  - Move the arrow handling out of `render_selection_left` (drop the `allow_nav` param) into that same fallback spot. `feed_keys` passing unbound single keys through to inpt is what makes this work, so keep it.
- `tui.boot_contexts` (set in `run_tui`) exists only for `da`. Forks and subagents already set `ctx.parent`, so "spawned" = `c.parent is not None`. Delete `boot_contexts`. Check that `inline.py` (~156) and `agents.py` (~222) contexts should count as spawned. Probably set parent there or accept it.
- `tui.set_favorite(ctx)` is a one-line setter. Use `tui.favorite = ctx` and delete the method. Update agents.py and AGENTS.md.
- `h` key forces work mode because ask panels only render in work mode (`_tui_loop` ~2729). Human note: "formalize the idea of human help so its usable in selection-mode". With `ask_human` in place, render `tui.current.ui_stack[-1]` in selection mode too. Then `h` just selects and the panel shows, and you answer without leaving the list.
- Dead `"help"` mode: in the `TUI.mode` Literal and an `assert` branch in `_tui_loop`, never set. Delete it.
- `Context.last_seen` is UI state written every work-mode frame. Fine, but `is_unseen` could live next to it. Low priority.
- `_o200k_cache` (hash -> count, cleared at 100k) works around selection mode recounting messages without `_current_tokens`. Check whether caching per Message (like `_render_cache`) is simpler, then delete the global.
- Yank keys (user decided): remove `ya` and `yl`. Keep `[n]yc` = yank Nth-last code block, add `[n]yr` = yank Nth-last response (agent message, what `ya` did). Update help text, AGENTS.md keys.py line, interact.md.
- interact.md STATUS block: once this lands, fold the status into interact.md, which stays the roadmap (registers, agent messaging, set_reasoning, undo).

## Files
- `_ex6/keys.py` -> `_ex6/vim.py`
- `_ex6/prompts.py`: `expand_abbrev` handler -> `transform_user_prompt`
- `_ex6/commands.py`: `/help`, `_text_panel`
- `_ex6/tools.py`: ask_user_question (~1217), other ui_stack tools (~1195, ~1353, ~1503)
- `_ex6/agents.py`: `set_favorite` call (~258)
- `ex6.py`: `Context.invoke` (~1338 pipeline), `push_ui` (~1455), `InputBox` (`_expand_abbrev` ~1856, draw ~1890-1935), `render_selection_left` (~2000), `transform_user_prompt`/`expand_abbrev` handleables (~2298), `TUI` (~2495: boot_contexts, favorite, set_favorite, mode Literal), `_tui_loop` (~2660-2731), `run_tui` (~617), `KeyArgs` (~106)
- `AGENTS.md`: decorators, core-plugins, ui-ux sections

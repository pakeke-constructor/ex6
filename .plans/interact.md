# interact: agents and users interact with ex6 better; vim-style UX

## Motivation
User's words: "What if agents could 'interact' with ex6 much better?" Clipboard, sending prompts to other agents, storing data in ex6 "like a buffer", settings. Broader: "What kinds of UX things would make ex6 easier to work with? Maybe prompts as a first-class primitive?"

Then: "I really wanna lean into the vim side of things." "Remove all the `/` commands, and replace them with keys." "Compound commands work too." `dd` pops last user message + copies it, `u` undoes (re-adds it), `f` forks, `t` "instantly brings you to workmode to the nearest agent that requires input".

The idea that ties it together: vim already has the right primitives. Registers = buffers/clipboard. `@a` = send a stored prompt. Undo = message history. Modes = selection vs work. No new concepts needed; the parts reinforce each other.

## Findings (from spike)
- Tools get `ctx` -> `ctx.app` -> `ctx.app.tui`. Most agent-facing features are plugin tools; no core change needed.
- Selection mode is already a vim normal mode: `j/k` nav, `/` opens command input, Enter enters work mode. Work mode: input always focused, ESC -> selection.
- `commands.py` smp/ssot/short are copy-paste "constant + optional note -> invoke". `sigils.py` re-describes the same ideas via a cheap LLM rewrite. User: "sigils should just be deleted asap, its shit" "the main important bit is the actual content like ssot and smp, thats only reason i keep the file". Delete first (Order step 0).
- `ctx.submit_input()` sets `stop_early = True` if target is running, so it interrupts. `invoke()` raises if running.
- Dynamic commands are possible: `app.commands[name] = (fn, spec)`.

## 1. Vim keys (core keymap)
Selection mode becomes normal mode. Keys act on `tui.current` (shown in right panel).

Core: `@ex6.key("dd")` registry next to `@ex6.command`, `fn(tui, count, register)`. Commands stay as the implementation; keys are bindings. Dispatch in `_tui_loop` selection branch.
- Parser: pending key buffer. Leading digits = count (`3dd`). `"x` prefix = register. Exact match with no longer binding -> run. Prefix of a binding -> wait. Else reset. No timeouts: never bind a key that is also a prefix (vim's `d` vs `dd` problem).
- `:` replaces `/` as the command line, for commands that take args (`:cm msg`, `:theme x`). Keep `/` in work-mode input during transition.
- `?` = help generated from keymap. which-key style popup after a prefix key (discoverability is critical once `/` commands go away).
- Drop `w/s` nav (collides with letters).

User's words: "z can be the key related to permissions. So zz toggles yolo." "yy yanks last user prompt. y[x] might be a modifier like yiw, yanks some of the agent's output." "There needs to be a way to navigate between agents, and to 'start' an agent besides enter."

Proposed keymap:

Navigate between agents:
- `j/k`, `gg/G`: next/prev, first/last in list.
- `gt` / `gT`: next/prev agent (vim tabs). `{n}gt`: jump to nth agent. Works from work-mode normal sub-mode too, so you can hop agents without going back to the list.
- `ctrl-^` (or `''`): back to previous agent (vim alternate buffer). Toggle between two agents you're juggling.
- `/name`: search agents by name, jump to match; `n`/`N` next/prev match. `/` is free once commands move to `:`.
- `t`: jump to work mode on next agent needing input. Pending question (`ctx.ui_stack` non-empty) first, then finished-and-unseen. Oldest waiting first. Show a marker in the selection list too.
- `m{a-z}` / `'{a-z}`: mark/jump agents.

Start agents (besides Enter):
- `i` / Enter: work mode, input focused.
- `a`: quick-prompt. One-line input at the bottom of selection mode, sends to selected agent, stays in selection mode. Lets you dispatch prompts to several agents (`a`, `j`, `a`, ...) without leaving the list.
- `o`: fork selected agent, then quick-prompt the fork (branch off).
- `r`: run/continue without a new message (= /go). `R`: retry: drop the last assistant turn and rerun.
- `@a`: send register a as prompt. `@@` repeat. `p`: work mode with unnamed register in input box.
- `.`: repeat last action (eg resend same prompt to another agent: `a ... <CR>`, `j`, `.`).
- `x`: stop (also ctrl-x).

Yank (`y` is an operator, takes an object like vim's `yiw`):
- `yy`: last user prompt (same as current /yy).
- `ym`: last assistant message. `yt`: last tool output.
- `` yi` ``: contents of last code block in last assistant message (vim "inside backticks"). `` ya` ``: same, with fences.
- Counts pick the Nth-last: `3yy` = 3rd-last prompt, `` 2yi` `` = 2nd-last code block.
- Registers: `"ayy` yanks into a. Default target is `"` + clipboard.

Edit history:
- `dd`: pop last user turn, put its text in `"` (+ clipboard). `3dd` pops 3 (= /pop n).
- `dd` then `p` = edit-and-resend last prompt.
- `u` / `ctrl-r`: undo/redo message history.
- `cc`: clear agent and enter input (vim "change").
- `f`: fork. `dc`: delete agent.

Permissions (`z` prefix):
- `zz`: toggle yolo.
- Later: other permission toggles under `z` (eg `zr` read-only: strip WRITE/SHELL tools). Never exposed to agents.

Settings:
- `>` / `<`: raise/lower reasoning of selected agent.

Undo: per-context stack of message-list snapshots, pushed by destructive keys (dd, cc, pop). Store on tui keyed by ctx (not `data_volatile`: clear wipes it and clear must be undoable); drop via `context_removed` hook. Refuse while ctx running.

Open: work-mode normal sub-mode. ESC from input -> normal (keys act, `j/k`/ctrl-d/u scroll), `i`/Enter -> insert, ESC again -> selection. Real vim feel, but changes ESC behaviour and ask panels also use ESC. Start with selection mode only.

## 2. Registers (replaces "clipboard tools" + "buffer")
One app-level store shared by user and agents. `+` = clipboard, `"` = unnamed, `a-z` named, `last` = last tool output / assistant msg. In `app.plugin_data` or volatile, optionally saved to `get_folder("registers")`. Not `ctx.data` (per-agent, scalars only).
- Tools: `reg_set(key, text)`, `reg_get(key)`, `reg_list()`.
- Token win: agent stashes big output under a key and passes the key to a subagent without the text going through its own context.
- In the input box, `;name` pastes register/prompt `name` inline (see section 3). `@file` attaches a file.
- Clipboard read by agents: risky (secrets). Write-only by default; read only via `_approve` with preview, never auto-approved under yolo.

## 3. Prompts as first-class
`_ex6/_prompts/*.md` with frontmatter, reusing `skills.py` `_parse_frontmatter`:
```
---
name: smp
description: step back, simplify
---
Take a step back...
```
User's words: "typing ;smp should paste the smp register." "remove the sigils system plugin, it's bad imo; and i havent found i like using it. take what you can from it with ssot and smp tho."
- Each prompt is a file-backed, read-only register named by filename. Replaces smp/ssot/short commands.
- `;name` in the input box pastes the prompt text in place, verbatim. No LLM rewrite (that was sigils' problem: it mangled user wording). User sees and can edit exactly what is sent.
  - Paste live in InputBox when `;name` + a non-word char is typed and `name` exists (vim abbreviation style). Unknown names stay as typed. Not a transform_user_prompt handler: hidden expansion on submit breaks "no hidden context".
  - Escaping: `;;name` -> literal `;name`. (`;;;` inline prompts live in files, not the input box; no clash.)
- `@` stays single-letter registers as a key; `;name` is the in-text form for any register, so the earlier `@smp` clash is gone.
- Pasted mid-sentence, so prompt files must read as standalone paragraphs.
- Prompt files already exist from Order step 0 (smp, ssot, short, git, fast, read, query); this section adds the loader + `;name` paste.
- `:save name` saves last user prompt as a prompt file.
- Skills and prompts: same format, opposite directions (agent pulls skill, user pushes prompt).

## 4. Agent -> agent messaging
- Tools: `list_agents()`, `message_agent(name, text, wait=False)`. `wait=True` behaves like explore_agent but on an existing context.
- Needs a non-interrupting send: core `ctx.send(text, sender)` that just queues (submit_input interrupts).
- Wrap incoming text in `<from agent="x">...</from>`.
- Escalation hole: only send to contexts with yolo <= sender's yolo, else ask approval.
- Ping-pong cost loops: hop counter in `ctx.data["msg:hops"]`, cap it.
- User-side: `:to opus review this` pipes current context's last assistant msg to another agent.

## 5. Settings
Never let agents change yolo, budget.limit, approval handlers. Only worth it: `set_reasoning(level)` for own context. Read-only spend/budget introspection is fine. No model switching (breaks cache, may switch backend).

## Order
0. ASAP, before anything else: kill sigils, keep its content.
   - Write `_ex6/_prompts/{smp,ssot,short,git,fast,read,query}.md` (frontmatter: name, description). Body: smp/ssot/short from commands.py SMP/SSOT_CHECK/SHORT (fuller than sigil wording); git/fast/read/query from the "Known sigils" list in sigils.py SYSTEM_PROMPT, expanded into standalone paragraphs.
   - Delete `_ex6/sigils.py` + its AGENTS.md line. No replacement needed yet; `/smp` `/ssot` `/short` keep working until step 3.
1. Keymap core + nav (`j/k`, `gt`, `ctrl-^`, `/`, `t`), start (`i`, `a`, `r`, `x`), `dd`/`p`/`u`, `yy`, `zz`, `f`, `:`, `?`.
   Then `y` objects, `o`, `R`, `.`, marks.
2. Registers (`"`, `+`, a-z, `@a`).
3. Prompts loader + `;name` paste; replace smp/ssot/short commands with the prompt files.
4. message_agent + `ctx.send` + yolo guard.
5. set_reasoning.

## Files
- `ex6.py`: `App.command`/`dispatch_command` (414-505), `TUI` (2356), `_tui_loop` selection branch (~2516), `render_selection_left` (1899), `Context.submit_input`/`invoke` (1117, 1243), `_try_clean_clipboard` (289)
- `_ex6/commands.py`: all `/` commands to rebind (clr, pop, yy, fork, stop, del_context, smp/ssot/short, help)
- `ex6.py` `InputBox.draw` (~1794): where `;name` paste hooks in
- `_ex6/sigils.py`: delete in step 0 (copy "Known sigils" content out first)
- `_ex6/skills.py`: `_parse_frontmatter`
- `_ex6/tools.py`: `_approve`, `ask_user_question` (uses `ctx.ui_stack`), `explore_agent` (wait pattern)



# HUMAN NOTE:
the following stuff is written by me, the human.

there is a branch called `vim` with a lot of this stuff implemented.
you might wanna take a look.

### MY OPINION, AFTER TRYING IT:
I think that the implementation in `vim` branch is WAYYY too similar to vim. it's actually kinda janky to use.

And in general; i think we can do it better.
we should work back from pain-points directly.
### IDEA:
- in work-mode, you are typing prompts. (like insert-mode in vim)
- in selection-mode, you do fancy keybinds. (like normal-mode in vim)
dont have any modes outside this.
HOWEVER: in selection-mode, we should ALSO be able to type prompts and dispatch them without opening work-mode.
IDEA: `space` is special character; if it's typed in selection-mode, the user opens input box for whatever agent is selected. Then when they press enter it is dispatched.

### COMPLETE LIST OF PRIMITIVES:
`h` => h stands for "human". Instantly opens the first ui-stack panel on the nearest agent that needs human help. (NOTE: might want to formalize the idea of "human help" so its usable in selection-mode?)
`p` => p pops a message
`g` => go. starts an LLM
`f` => fork
`j k` => up-down
`[n]yy` => yank Nth last user prompt
`[n]yc` => yank agent's Nth last code-block
`[n]yl` => yank agent's last N lines of output (defaults to 1 line)
`[n]ya` => yank agent's Nth last code-block
`[n]dd` => delete last usr prompt
`da` => delete ALL spawned agents (agents that ex6 didnt start with)
`n` => creates a new agent and opens input for typing (in selection-mode). Configure your favorite agent with `tui.set_favorite(agent)`
`cc` => pops last user prompt and opens input
`ca` => pops last user prompt, opens input, and pastes last user prompt. Cursor it at end of line

### Compound stuff:
`f<space>hello there<enter>` => forks a context and types hello
`ohello there<enter>` => (o is same as `f<space>`) forks a context and types hello

- For the ask-user-question stuff, make the UX WAYYY better. should be able to tap buttons.



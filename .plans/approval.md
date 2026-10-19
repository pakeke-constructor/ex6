# approval

## Motivation
User: "formalize the approval / permissions system a bit... but something still simple."
Before: `ctx.yolo` bool + each tool calls `approve()` in tools.py with its own panel/preview.
Goals (user's words):
- allow subagents/special classifiers to auto-approve or deny tools
- allow user to fully control approval pipeline
- allow different approvals to exist per tool

Fits ex6 philosophy: no hidden policy in core; approval is just a handleable, plugins decide everything.

## Core idea: ordered chain, first verdict wins (not OR, not AND)
Each handler returns one of three things:
- `True`        -> allow
- `"reason"`    -> deny (string goes back to LLM)
- `None`        -> abstain, ask next handler

First non-None verdict wins. If all abstain, `request_approval` returns None and the caller decides (tools.py shows the human panel). Denials are final - never fall back to the panel.
This is exactly what `App.call()` already does (handlers in `order`, first truthy wins, fallback default). No new combinator needed.

Why not OR / AND:
- OR (any approve -> approved): one permissive plugin silently overrides a safety deny.
- AND (any deny -> denied): can't auto-approve unless every handler agrees; human can't be a member of the vote.
- Ordered chain: priority is explicit via `order=`. Hard denies at order -100, classifiers at 0, yolo at 100, human panel = caller fallback when undecided.

## Core API (ex6.py) - implemented
Core knows nothing about UI. Lives in core so the handleable exists before any plugin loads; plugins use module-level `@ex6.handler`.

    verdict = ex6.request_approval(ctx, tool, args)   # True = approved, "reason" = denied, None = no handler decided

- `args` = the tool's raw args. No title/preview/diff.
- `approve_tool(ctx, tool, args)` handleable; default returns None.
- yolo: core `_approve_yolo` registered at order 100 in `App.__init__` (different Python name so it doesn't shadow the handleable). `/yolo` command is in core too.
- Handleables/overridables are core-only: decorators are private `_handleable` / `_overridable`.

## Human panel (_ex6/tools.py)
- `_ask_human(ctx, title, preview=None, height=None)`: plain function, not a handler. No `ctx.app.tui` -> approve. `height` = short bottom panel (shell tools).
- `_approve(ctx, tool, args, title, preview, height)`: `request_approval`, then `_ask_human` if None. Returns None if approved, else denial reason.
- Each tool builds its own preview (diff / command box) from its raw args.

## Example handlers (plugins, not core)
- read-only tools: `if tool in READ_TOOLS: return True`
- hard deny: `if tool == "powershell" and "rm -rf" in args["command"]: return "blocked"`
- path guard: deny writes outside `ctx.cwd`
- classifier subagent: run small LLM on req, return True / reason / None (unsure -> falls to human)
- per-context: subagent preset sets a ctx field, handler reads it (e.g. explore agent = read-only)

## Alternatives considered
1. Deny-overrides (AWS IAM style): run ALL handlers; any deny -> deny, else any allow -> allow, else human. Order-independent, safer when plugins don't know about each other. Needs a new combinator in core (~10 lines). Good fallback if chain ordering gets confusing.
2. Declarative rule table: `ctx.rules = [("powershell", "git *", "allow"), ("*", "*", "ask")]`. Nice for users, but just one handler on top of the chain - build later if wanted.
3. Effect tags on tools: tools declare `effects={"read"|"write"|"exec"|"net"}`; policies/modes over effects, not names (plan-mode = read only). Fewer rules, but touches every tool def. Could add an `effects` arg later.
4. Policy fn on Context: `ctx.approver = fn`. Simplest, per-context by nature, but only one policy - no composition.

## Nice-to-have (not v1)
- Human panel option "always allow for this ctx" -> appends rule to ctx.
- Show which handler decided (log line in conversation: "auto-approved by classifier").
- Classifier note passed through to human panel when it abstains.

## Files
- `ex6.py`: `request_approval`, `approve_tool`, `_approve_yolo` (~2040), `/yolo` (near `/dbg`), `App.__init__`, `App.call()`
- `_ex6/tools.py`: `_ask_human`, `_approve`, `_approve_command`; call sites in write_file, patch_file, edit_file, edit_file_lines, bash, powershell

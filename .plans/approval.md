# approval

## Motivation
User: "formalize the approval / permissions system a bit... but something still simple."
Today: `ctx.yolo` bool + each tool calls `approve()` in tools.py with its own panel/preview.
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

First non-None verdict wins. If all abstain, default implementation runs = the human approval panel.
This is exactly what `App.call()` already does (handlers in `order`, first truthy wins, fallback default). Zero core changes.

Why not OR / AND:
- OR (any approve -> approved): one permissive plugin silently overrides a safety deny.
- AND (any deny -> denied): can't auto-approve unless every handler agrees; human can't be a member of the vote.
- Ordered chain: priority is explicit via `order=`. Hard denies at order -100, classifiers at 0, yolo at 100, human = fallback.

## Request object
Tools stop building panels. They build a request and call one function:

    req = ApprovalRequest(ctx, tool="edit_file", args={...}, title="Edit file: x.py", preview=render_fn)
    denial = request_approval(req)   # None = approved, str = denial reason

`request_approval` (tools.py):
- `ctx.stop_early` -> "stopped"
- `v = ctx.app.call("approve_tool", req)`
- normalize: True -> None, str -> str

`@ex6.handleable def approve_tool(req)` default = current panel UI (from `approve()`), using `req.title` + `req.preview`.
Per-tool behaviour = handlers filter on `req.tool` / `req.args`. No per-tool registry needed.

## yolo becomes a handler
    @ex6.handler(order=100)
    def approve_tool(req):
        if req.ctx.yolo: return True
Keep `ctx.yolo` field for now (agents_long_horizon sets it). Later maybe `ctx.approval_mode: str` shown in footer (ties into footer "modes" todo in ex6.py ~2047).

## Example handlers (plugins, not core)
- read-only tools: `if req.tool in READ_TOOLS: return True`
- hard deny: `if req.tool == "powershell" and "rm -rf" in req.args["command"]: return "blocked"`
- path guard: deny writes outside `ctx.cwd`
- classifier subagent: run small LLM on req, return True / reason / None (unsure -> falls to human)
- per-context: subagent preset sets a ctx field, handler reads it (e.g. explore agent = read-only)

## Alternatives considered
1. Deny-overrides (AWS IAM style): run ALL handlers; any deny -> deny, else any allow -> allow, else human. Order-independent, safer when plugins don't know about each other. Needs a new combinator in core (~10 lines). Good fallback if chain ordering gets confusing.
2. Declarative rule table: `ctx.rules = [("powershell", "git *", "allow"), ("*", "*", "ask")]`. Nice for users, but just one handler on top of the chain - build later if wanted.
3. Effect tags on tools: tools declare `effects={"read"|"write"|"exec"|"net"}`; policies/modes over effects, not names (plan-mode = read only). Fewer rules, but touches every tool def. Could add `req.effects` later.
4. Policy fn on Context: `ctx.approver = fn`. Simplest, per-context by nature, but only one policy - no composition.

## Nice-to-have (not v1)
- Human panel option "always allow for this ctx" -> appends rule to ctx.
- Show which handler decided (log line in conversation: "auto-approved by classifier").
- Classifier note passed through to human panel when it abstains.

## Files
- `_ex6/tools.py`: `approve()` (~1387), `_approve_command()` (~1496), call sites at ~270, 384, 417, 540, 1585, 1610
- `ex6.py`: `App.call()` (~448), `handler` decorator (~85), `Context.yolo` (~972), footer (~2034)
- `_ex6/commands.py`: `/yolo` (~86)
- `_ex6/provider_anthropic.py`, `_ex6/provider_openai.py`: footer overrides reading `ctx.yolo`

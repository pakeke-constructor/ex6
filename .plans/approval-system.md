# Formal approval / permissions

## Motivation

Replace scattered tool-owned approval behavior and `yolo: bool` with one small, explicit protocol. Plugins, subagents, and classifiers should be able to approve or deny an operation; users should own policy; each tool can supply its own approval data and presentation without embedding approval control flow.

This is policy coordination between trusted plugins, not a security sandbox: plugins already execute arbitrary Python.

## Recommendation: one overridable decision function

Avoid baking OR/AND voting into core. Expose one overridable approval function; whoever owns the override owns composition and conflict semantics.

Conceptual API:

```py
ApprovalRequest(
    tool="write_file",
    action="write",
    description="Write file: ex6.py",
    arguments={...},
    preview=...,       # optional UI/render data
)

ApprovalDecision.approve()
ApprovalDecision.deny(reason)
ApprovalDecision.ask()

request_approval(ctx, request) -> ApprovalDecision
```

Default policy:

1. Stop/cancel -> deny.
2. Allow-all mode (current `yolo`) -> approve.
3. Otherwise -> `ask`, resolved by standard human approval UI.

An override can:

- approve known-safe calls;
- deny known-dangerous calls with a reason;
- invoke a classifier/subagent;
- branch on `request.tool`, action, arguments, context name, or `ctx.parent`;
- delegate undecided requests to default human behavior.

This is simpler than a multi-handler voting protocol, has one obvious policy owner, and gives the user full control. A user who wants several classifiers can compose them inside the override.

## If composable handlers are desired

Use an ordered, first-decisive chain rather than global OR aggregation:

```text
handler -> APPROVE | DENY(reason) | ASK | ABSTAIN
```

- Handlers run in explicit priority order.
- First result other than `ABSTAIN` wins.
- If all abstain, default human prompt runs.
- `ASK` immediately routes to human approval and prevents a later permissive handler bypassing it.

This lets the user express either approve-first or deny-first policy through ordering. It also makes conflicts deterministic and debuggable.

Do not use approve-wins OR as the core rule. One permissive or buggy plugin could bypass every denial. Do not use deny-wins as a fixed core rule either; it prevents the user from intentionally placing a trusted approval policy ahead of conservative classifiers. If classifiers should vote in parallel, implement that voting rule as one user-owned handler/override.

## Separation of concerns

- Tool prepares an `ApprovalRequest` containing operation-specific facts and optional preview.
- Approval policy decides approve / deny / ask.
- Standard UI resolves `ask` and is the only code that opens the approval panel.
- Tool executes only after approval and converts denial into its normal tool error result.

Tools should no longer directly push panels or know keyboard semantics. File tools may provide diff preview data; shell tools may provide command text. UI remains generic and can render optional request-specific preview.

Keep approval requests explicit inside mutating/dangerous tools rather than intercepting every call in `call_tools`. Central interception lacks computed diffs and would require a tool metadata/decorator system. Since plugins are trusted, explicit calls are the smaller design. A later tool decorator can be added only if missing approval calls become a real problem.

## Per-tool policy

Use `request.tool` plus a small `action` string, not a registry of bespoke callbacks per tool.

Examples:

- `write_file`: tool=`write_file`, action=`write`, arguments include path, preview is diff.
- `edit_file`: tool=`edit_file`, action=`edit`, arguments include path/search, preview is diff.
- `powershell`: tool=`powershell`, action=`execute`, arguments include command/shell.

Policy can branch directly:

```text
read-only tools                  -> approve
write_file under workspace/**   -> approve
powershell running tests         -> approve
powershell deleting files        -> deny
anything else                    -> ask user
```

Avoid a capability hierarchy (`fs.write`, `shell.exec`, risk levels) initially. Add tags later only if repeated tool-name checks become painful.

## Classifier/subagent behavior

Classifier receives a serializable view of request, not renderer callbacks or mutable internals. It returns only approve, deny(reason), or abstain/ask.

Suggested safeguards kept in user policy rather than core:

- classifier timeout -> ask human;
- malformed response -> ask human;
- classifier denial reason is surfaced to calling model and human;
- cache/scope approvals only if user explicitly implements it.

Do not silently treat classifier failure as approval.

## `yolo` migration

Treat `yolo` as the built-in allow-all policy, not a special check inside every tool. Initially it can remain context configuration consumed by default approval policy. Longer term replace the boolean with a named policy/mode such as `ask` / `allow_all`, and have footer display active policy rather than only `yolo ON/OFF`.

## Relevant files

- `ex6.py`
  - approval request/decision types and overridable dispatch point;
  - `Context` policy/mode state;
  - work-mode footer currently showing `yolo`.
- `_ex6/tools.py`
  - current `approve` panel;
  - file mutation approval calls;
  - `_approve_command`, `bash`, and `powershell`;
  - operation-specific preview construction.
- `_ex6/commands.py`
  - any command that currently toggles or exposes `yolo`.

## Implementation sequence

1. Define exact request and decision shapes. Keep fields minimal: tool, action, description, arguments, optional preview.
2. Add one overridable `request_approval(ctx, request)` with default allow-all-or-human behavior.
3. Move panel/input behavior into default human resolver.
4. Convert file mutation and shell tools to construct requests and call the shared API.
5. Replace tool-specific denial wording with one consistent denial result carrying reason.
6. Update footer/commands to describe active approval mode.
7. Add focused tests for approve, deny, ask fallback, per-tool branching, cancellation, and classifier failure fallback.
8. Verify file diff and shell command previews remain unchanged from user perspective.

## Deferred ideas

Only add these after concrete need:

- persistent "approve this once/session/path" scopes;
- capability tags or risk levels;
- quorum/parallel voting;
- audit log of decisions and policy source;
- call-site enforcement in `call_tools`;
- asynchronous approval protocol.

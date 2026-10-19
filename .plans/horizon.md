# horizon: super basic overnight task loop

## Motivation
"Long running agents that run all-night-long." Human writes all tasks in a file, goes to bed, ex6 works through them one by one.
Each task gets its own branch, a review pass, a docs pass, then gets merged + archived.
Everything the agents did goes into an audit folder, so "the human operator [can] audit EVERY change that happened, and see exactly what went wrong and why."
Super basic version first. Keep custom agent sys-prompts "deliberately simple and short".

<original-user-spec>
horizon plugin:
Long running agents that run all-night-long.

```py
# human; writes all tasks in a file.
process_all_tasks(task_agent)

while tasks:
    make_new_git_branch() # from master

    t = tasks.peek()
    do_task(main_agent, t, can_loop=True)
    if review(review_agent) # ssot, conciseness, simplicity
        do_task(main_agent, t, can_loop=False)

    update_docs(doc_agent, t) # updates agents.md

    finalize_task(t, ...)


"""
throughout the task, the agent uses a folder called `workspace/**`.

It uses this folder to put ALL of it's screenshots, outputs, and especially any of it's temp files that it uses to test stuff / smoke-tests.

The agent doesn't need to delete anything from workspace; it should keep everything for auditability.
This gives the system incredibly good observability.

-----------------------------

For the `tasks.peek()` and `tasks.pop()` stuff:
it should NOT use an actual in-memory list.
instead, it should

"""

def finalize_task(t):
    """
    - merges the current branch into master.
    - pops tasks from the json list.
    - creates a new folder, called `ex6.history/{timestamp}_{commit_id}/**. (Lets call this folder the `audit` folder for brevity.)
    - moves EVERYTHING from workspace/** into audit/workspace/**
    - copies a list of agent-context windows into audit. So you get `main_agent.0.txt`, main_agent.1.txt`, etc etc. `review_agent.0.txt`, `doc_agent.0.txt`.
    - This will allow the human operator to audit EVERY change that happened, and see exactly what went wrong and why.
    """
```

</original-user-spec>


## Loop (from spec)
```py
while tasks:
    make_new_git_branch()            # from master
    t = tasks.peek()
    do_task(main_agent, t)
    if review(review_agent):         # ssot, conciseness, simplicity
        fix_review(main_agent, feedback)
    update_docs(doc_agent, t)        # updates AGENTS.md
    finalize_task(t)
```

## Dependency: tasklist
Queue = markdown file, NOT in-memory list. See `.plans/tasklist.md`. Implemented in `_ex6/agents_long_horizon.py`: `split_tasks(text) -> (preamble, tasks)`, `join_tasks(preamble, tasks)`. Tasks = raw strings, heading included.
Loop does its own file IO: peek = `tasks[0]`, pop = re-read file, drop `tasks[0]`, write back. Task file lives outside repo or gitignored (branch switching).

## Design
New file `_ex6/horizon.py`. Imports `TASK_SYSTEM_PROMPT`, `MAIN_TOOLS` from `agents_long_horizon.py` and presets from `agents.py`.

Entry: `run_horizon(app, tasks_path, main=SOL, review=SOL, doc=SOL)`. Each agent takes a preset dict from `agents.py` (SONNET, OPUS, SOL, TERRA, ...), e.g. `Context(app, **{**preset, "name": "h_main"}, messages=...)`.
`/horizon <tasks_path>` command calls it with defaults (SOL for all, for now). Runs loop in a background thread (Context.invoke is async).
3 contexts added to tui so user can watch live: `h_main`, `h_review`, `h_doc`. Each is `clear()`ed at start of every task.

Blocking helper:
- `run(ctx, prompt)`: `ctx.invoke(prompt)`, then poll until done.
- Done = `not ctx.is_running()` AND last message is assistant with no tool_calls.
  Reason: `handoff` clears + re-invokes via `schedule`, and there is a short window in `invoke`'s `finally` where `is_running()` is False but the scheduled re-invoke hasn't happened yet. Checking last message avoids that race.
- If `ctx.stop_early` -> raise, abort loop (user hit /stop).

Phases:
- make_new_git_branch: refuse to start if `git status --porcelain` dirty. `git checkout master`, `git checkout -b horizon/{timestamp}`.
- do_task: prompt = task title + body + workspace rules. Main tools incl. `handoff`.
- review: read-only tools + COMMANDLINE_TOOL (for `git diff master`). Prompt: check SSOT, conciseness, simplicity. Last line must be `VERDICT: OK` or `VERDICT: CHANGES`. Returns (needs_changes, review_text).
- fix_review: same h_main context (keeps its history), prompt = review feedback; "address this feedback only, then stop." No second review. (No `can_loop` flag; the `if review` branch is the only control flow.)
- update_docs: doc agent gets `git diff master` + task; updates AGENTS.md only if something worth documenting.
- finalize_task:
  1. `git add -A`, `git commit -m "horizon: {title}"`, `git checkout master`, `git merge --no-ff <branch>`.
  2. pop task (re-read file, drop first, write).
  3. audit = `ex6.history/{timestamp}_{short_commit_id}/`.
  4. move `workspace/**` -> `audit/workspace/**`.
  5. write context dumps: `main_agent.0.txt`, `main_agent.1.txt`, `review_agent.0.txt`, `doc_agent.0.txt`, plus `task.md` (title+body).
  6. `git add ex6.history`, `git commit -m "horizon audit: {title}"` on master. (Audit is committed: user wants backups; a cleanup script for old histories can come later. Separate commit because folder name needs the merge commit id.)
- Any git failure (merge conflict etc): stop loop, leave repo as-is for human.

Context dumps (`.N` = one file per context window):
- A window ends on `handoff` (context gets cleared) or at end of task.
- horizon passes its own `handoff` tool (functools.wraps of tools.handoff) that dumps the ctx text into a per-task list before delegating. End of task dumps the final window.
- Extract dump formatting out of `/dmp` in `commands.py` into `format_context(ctx) -> str`; `/dmp` and horizon both use it.

Workspace:
- All agents told: put ALL screenshots, outputs, temp/smoke-test files in `workspace/`. Never delete from it.
- `workspace/` added to `.gitignore` (else `git add -A` commits it, and branch switching moves it around).
- `ex6.history/` is committed (only on master, via the audit commit).
- Risk: agents' glob/search may hit `ex6.history/` and get noise. Tell agents to ignore it; revisit if it's a problem.

## Sys-prompts (draft, keep short)
main: TASK_SYSTEM_PROMPT + 
"You are running unattended; no human will answer questions. Make reasonable decisions and note them in your final message.
Put ALL temp files, smoke tests, screenshots and outputs in `workspace/`. Never delete anything in `workspace/`. Don't commit; the harness commits."

review:
"You review a diff (`git diff master`) for one task. Do not edit code.
Check: single-source-of-truth (no duplicated/fragile state), conciseness, simplicity, task actually done.
Give short concrete feedback. Final line exactly `VERDICT: OK` or `VERDICT: CHANGES`."

doc:
"You maintain AGENTS.md. Given a task and its diff, update AGENTS.md only if a future agent needs to know something new (new module, convention, gotcha). Keep it short. If nothing to add, change nothing."

## Out of scope for v1
- `process_all_tasks(task_agent)` (pre-processing the task file). Human writes tasks directly.
- Retry loops, parallel tasks, worktrees, budget caps beyond existing global spend limit.


## Files
- `_ex6/horizon.py` (new, ~150-200 lines)
- `_ex6/agents_long_horizon.py` (TASK_SYSTEM_PROMPT, MAIN_TOOLS, tasklist API)
- `_ex6/commands.py` (extract `format_context` from `/dmp`)
- `_ex6/tools.py` (`handoff`, `COMMANDLINE_TOOL`, `ENV_PROMPT`, `AGENTS_MD`)
- `ex6.py` `Context.invoke` / `schedule` (handoff race context)
- `.gitignore`
- `_notes/TASKS_TODO_PROMPTS.md` (original spec)

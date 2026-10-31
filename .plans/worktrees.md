# worktrees: git worktrees for ex6 agents

## Motivation
User: "Let's get git-worktrees set up for ex6 agents." Multiple agents editing the same checkout trample each other (and the user). A worktree per agent = isolated branch + dir, merged back when done.
User's open questions:
- "where are git worktrees stored? (in ex6.get_folder("projectname/worktrees") maybe?) Or simpler: in the project itself, like .ex6.worktrees/**?"
- "can worktrees be cleaned up automatically after some time? (E.g. if all worktrees folder in the project sum to more than 500mb AND there are ones that havent been used in a while => performs cleanup step?)"
- "how are agents made aware that they are in a worktree? Because ideally they'll want to merge then they are done, right"

## Key finding: ctx.cwd already does most of the work
Every core tool resolves via `ctx.cwd or os.getcwd()`: read/edit/glob/search (`ctx.resolve`), bash/powershell (`cwd=ctx.cwd`), git diff tool, AGENTS_MD, ENV_PROMPT (already shows cwd + git branch), tasks.py `.plans/`, provider_anthropic spawns `claude -p` with `cwd=ctx.cwd`. So "agent in worktree" == `ctx.cwd = <worktree path>`. No core change needed.
Gaps (use `os.getcwd()`, not ctx.cwd): skills.py `_skills_dir`, inline.py, agents.py EX6 prompt. All fine to leave: they're about the main project, not the agent's files.

## Design (new plugin `_ex6/worktrees.py`, target <200 lines)

Storage: in-project `.ex6.worktrees/<name>/`, branch `ex6/<name>`. (Recommend over get_folder.)
- Matches existing `.ex6.history/` convention; same drive (no cross-volume issues on Windows); easy to find/inspect; per-project size check is just one dir.
- Must be gitignored. glob/search already skip gitignored paths. `ex6 init` (ex6.py ~2302) should add `.ex6.worktrees/` to .gitignore and pyrightconfig `exclude`, same as context dump.
- Alternative if user prefers: `get_folder("worktrees")/<project-hash>/<name>`. Only the root path constant changes.

API (for project plugins building agents):
- `attach(ctx, base=None) -> path`: `git worktree add -b ex6/<name> .ex6.worktrees/<name> <base or current branch>`, set `ctx.cwd`, `ctx.data["worktree:base"] = base`, append WORKTREE_PROMPT system msg. Name = ctx name slug + short id (collisions on forks).
- Copy untracked essentials into new worktree: `WORKTREE_COPY = [".env"]`. (.venv not needed: subprocesses inherit env/abs paths.)
- Forks share the parent's worktree (cwd is copied by `_clone`). Want isolation -> call `attach(fork)` explicitly.
- Must be called before first invoke. provider_anthropic's long-lived proc is spawned with cwd; add a small check there: if `ctx.cwd` != spawn cwd, restart proc.

Agent awareness: WORKTREE_PROMPT, dynamic system Message (callable content):
"You are working in a git worktree at {cwd}, branch ex6/<name>, based on <base>. The user's main checkout is elsewhere; don't touch it. Commit as you go. When the task is done and committed, call `merge_worktree`."
Tool `merge_worktree(ctx)`:
- refuse if worktree dirty (tell agent to commit).
- in main repo: refuse if HEAD isn't on base, or main tree has dirty changes overlapping the diff (git refuses anyway; surface its message).
- `git merge --no-ff ex6/<name>` in main repo. On conflict: `git merge --abort`, tell agent "run `git merge <base>` in your worktree, resolve conflicts, commit, retry".
- goes through approve_tool, so the user gates merges unless yolo. That's the human checkpoint.
- after success: worktree stays (agent may continue); cleanup handles it later.

Commands:
- `/wt` - attach current ctx to new worktree (refuse if running / has assistant msgs, per anthropic proc note; or just restart proc).
- `/wts` - list worktrees: name, branch, merged?, dirty?, last used, size.

Cleanup (on boot, background thread; also on `context_removed`):
- Last-used: touch `.ex6.worktrees/<name>.used` marker in an `after_llm_turn` hook. Dir mtime is unreliable.
- `git worktree prune` first (handles dirs deleted by hand).
- If total size of `.ex6.worktrees/` > 500MB: for worktrees unused > 7 days and not attached to a live context (oldest first, until under limit):
  - dirty -> auto-commit "ex6: wip (auto-cleanup)" so no work is lost.
  - `git worktree remove --force`.
  - delete branch only if merged into base; unmerged branches stay (commits are cheap, work recoverable).
- `context_removed`: if worktree is clean AND merged -> remove worktree + branch immediately. Otherwise leave it.
- Size walk can be slow (node_modules) -> only at boot, in thread.
- Constants at top: `CLEANUP_MB = 500`, `STALE_DAYS = 7`.

## Follow-ups (not in this plan)
- agents_long_horizon could run tasks in a worktree so the user's checkout is never touched (currently `_git` runs in os.getcwd and checks out branches there).
- Selection-mode right panel / work footer showing branch tag.

## Relevant files
- ex6.py: `Context.cwd` (~1049), `resolve` (~1170), `_clone` (~1372), `remove_context`/context_removed (~2407), init/gitignore (~2302)
- _ex6/tools.py: `_env_content`, `_is_gitignored`, git diff tool (~1731)
- _ex6/provider_anthropic.py: proc spawn (~277), context_removed (~118)
- _ex6/agents.py: where agents are built (call `attach` there or in project plugins)
- _ex6/agents_long_horizon.py: `check_git_state`, `_git` (follow-up)

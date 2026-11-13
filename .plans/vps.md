# vps: run plans on a VPS (docker per job), via /plans
Status: active

## Motivation
"man, i wish i could offload this agent to a VPS." ex6 over ssh/tmux has bad responsiveness, so the TUI stays local and the work moves.
The unit of work is a plan, not an agent. Don't copy context windows: a plan is the compressed version of a conversation, fresh context is better for long-horizon, and it avoids all porting issues (Windows paths, powershell, provider session formats, callable messages).
Send off and forget. The remote agent never pushes to master/dev, only to its own branch, which the TUI becomes aware of via syncing (SSOT).
This replaces the parked remote.md (live client-server sync), which was too complicated for ex6.

User, verbatim:
> Yeah we should remove the vps screen, reuse the plans system. This makes stuff way easier to use as well. altho I would like useful tools like being able to get git diff links on github easily/quickly, and having easy tools to "auto merge" all branches or something.

Decisions:
- One VPS, one throwaway docker container per job (`docker run --rm`).
- Subscriptions must run on the VPS (claude + codex). Crucial.
- No separate vps screen: `/plans` is the dashboard for local and remote work.
- Git worktrees must get cleaned up properly.

## The plan file is the channel
Plan rules already describe progress: finished parts deleted, remaining work kept, Status updated. On the job branch:
- plan deleted: done
- plan rewritten, `Status: active`: partial
- `Status: blocked on <question>`: stuck. Answer or add follow-ups by editing the plan and re-sending.

## VPS layout
- `/srv/ex6/repos/<project>.git`: bare repo per project. Local git remote `vps`.
- `/srv/ex6/jobs/<id>/`: job metadata, log, remote transcript (debug only, never imported).
- A small host script (`run-job`, `stop-job`, `ls-jobs`) starts containers and fetches each container's HEAD into `ex6/vps/<plan>` every minute and on exit. Containers can't touch the bare repo, so only that branch is ever written.
- Container: clones the branch, runs `run_horizon` on the plan (run -> review -> fix_review), long_horizon + yolo, commits after each turn. The last commit message carries the summary.

## Subscription auth in containers
- Claude: `claude setup-token` once, long-lived token passed as env.
- Codex: refresh tokens are single-use. Bind-mount one shared host auth.json into every container, so the existing `fcntl.flock` refresh lock works across containers. Never copy it per container.

## /plans UX
- Row markers: `●` local agent, `^` running remotely, plus done / blocked / dead (the container is gone with no done/blocked plan state). State comes from one `git fetch vps` plus `ls-jobs` over ssh.
- `v`: send the plan. Commit it, push `ex6/vps/<plan>` to `vps`, `ssh vps run-job`. Also push the branch to origin (GitHub) so diff links work.
- `g` on a finished remote plan: pull. Fetch the branch, create a local worktree on it (worktrees.py), and spawn a briefing agent with the original plan, the final plan, and `git diff base...branch`. Its first message: what changed, what's left, what's risky.
- Diff link: a key that yanks/opens `https://github.com/<owner>/<repo>/compare/<base>...ex6/vps/<plan>`.
- Auto merge: a key that merges every done branch into its base, one by one. On conflicts, skip that branch and report it (or hand it to an agent to resolve in its worktree). Never force anything.
- `s` stop the container, `dd` delete the job + branch, `l` show the job log.
- `/vps` on an agent: hand off to a plan (handoff tool), then send it like `v`.

## Cleanup
- Container: deleted on exit (`--rm`).
- VPS job folder + branch: deleted on pull or `dd`, and auto after N days if never pulled.
- Local worktree: only exists while the work is local (recreated on pull). After merge, the existing worktrees.py rules remove it and its branch.

## Open
- Cross-project view: `/plans` is per project. Maybe a flag to list remote jobs from all projects.

## Relevant files
- _ex6/plans.py: /plans panel (`_load_plans`, `_spawn`, `_panel`), keys g/a/i/t/dd
- _ex6/agents_long_horizon.py: `run_horizon`, `default_pipeline`, `check_git_state`
- _ex6/worktrees.py: `attach`, `_remove`, `cleanup`, `ex6/<name>` branches
- _ex6/tools.py: handoff tool; ask_user_question raises under `app.long_horizon`
- _ex6/provider_openai.py: `_refresh_codex_token` (flock, single-use refresh)
- _ex6/provider_anthropic.py: spawns `claude` (needs auth in container)

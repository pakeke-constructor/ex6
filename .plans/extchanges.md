# extchanges: tell agents when files they've read were changed under them
Status: parked (proposal; waiting on user's go-ahead and the hook decision below)

## Motivation
When files an agent has already read get changed by someone else (the user, bash, another agent), tell the agent automatically and show it the diff.

What the logs show (invokes2.db, Nov 10-13, about 4.5k invokes):
- About 28 of the user's roughly 1,460 prompts exist only to tell the agent about edits: "i made some changes", "ive made some adjustments", "re-read the plan. ive done a tonne of fixes", or `;git` / "check the git working tree".
- 9 `edit_file` calls failed because the file had changed since the agent read it. Each failure cost an extra round trip at about 60k of context.
- Even when the agent knows a file changed, it can't see what changed, so it re-reads the whole file. There were 433 `read_file` calls.
- ex6 already records a hash of every file an agent reads or writes (`ctx._read_hashes`), but only uses it to refuse stale edits (`_check_stale`). It never tells the agent what happened.

## The feature
When a file the agent has read or written has changed since, add a `<files_changed>` block containing a unified diff, then count the file as freshly read.
- On the user's turn: the block is added to the user's message.
- While the agent is running: an after_tool_calls hook adds it to the latest tool result, the same way `reminders.py` adds its notes. That catches the user's edits made mid-run before they turn into failed edits.
- Large changes: the diff is capped at about 40 lines per file. Past that, or if the file was deleted, the block just says the file changed and must be re-read, and edits stay blocked until it is.
- The block shows up in the transcript, so nothing is hidden from the user.
- Size: about 80 lines. ex6.py keeps each file's contents instead of only a hash (`mark_file_read`), compares them in `file_changed_since_read`, and adds `ctx.changed_files() -> [(path, old, new|None)]`. A new `_ex6/file_changes.py` builds the diff and adds the hook.

## Open decision
Adding the block to the user's message needs a hook that runs right before the prompt is sent. Either:
- Add a small `before_invoke` hook to core (preferred).
- Reuse `transform_user_prompt`. No core change, but the user has to press Enter twice.

## Files
- ex6.py (`mark_file_read`, `file_changed_since_read`, `invoke`)
- _ex6/tools.py (`_check_stale`)
- _ex6/reminders.py (pattern for after_tool_calls appends)

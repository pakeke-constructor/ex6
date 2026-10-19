# tasklist: markdown task-file format + read/write API

## Motivation
Long-running overnight agent loop (see `_notes/TASKS_TODO_PROMPTS.md`) needs a task queue: `tasks.peek()` / `tasks.pop()`.
It "should NOT use an actual in-memory list"; the queue lives in a file.
JSON is ugly. Use ONE markdown file with a `##` heading per task.
**Written by human, read and interacted with by agents.**
Scope: "all i want is api for reading/writing these file types." No tools, commands, or loop logic here.

## Format
````md
Optional preamble. Free text before the first heading.
Shared rules/context for every task.

## Fix scroll jitter
Body of task. Any markdown.
```py
## this is a python comment, NOT a new task (inside fence)
```

## Add /history command
...
````
- A task = a `## ` line at column 0, **outside ``` fences**, plus everything up to the next such line.
- Title = heading text after `## `. Body = raw text verbatim (no stripping/reformatting).
- `#` and `###`+ are just body text, not task boundaries.
- Preamble = text before the first task. Kept verbatim on rewrite.
- File order = queue order. First task = next task.
- Untouched tasks must round-trip byte-for-byte (human's formatting is sacred).

## API
DONE, in `_ex6/agents_long_horizon.py`. User: "All we need [is] tasks_to_string and string_to_tasks."
- `string_to_tasks(text) -> (preamble, tasks: list[str])`: fence-aware split. Each task = raw text, heading line included.
- `tasks_to_string(preamble, tasks) -> str`: inverse. `tasks_to_string(*string_to_tasks(t)) == t`.
Callers do file IO. When popping, re-read the file fresh first (human may have edited during a multi-hour task).

That's all. No locking, no ids, no status fields. Done = popped (and archived by the caller into the audit folder).

## Robustness rules (from discussion)
1. Fence-aware splitting. `## ` inside ``` blocks is body text.
2. Atomic writes via tmp + `os.replace`.
3. `pop` re-reads before writing.
4. Task file must NOT be git-tracked. The loop switches branches and merges, and a tracked file would change/conflict per branch. The API just takes a path; the caller keeps it gitignored or outside the repo.
5. Agents with `edit_file` could touch it. Outside the repo solves this as well.

## Testing
Small pure test for `parse`/`render`: round-trip, fenced `##`, preamble, empty file, no trailing newline.
`peek`/`pop`/`append`: temporary `smoke_test.py` against a tmp file, then delete it.

## Files
- `_ex6/agents_long_horizon.py`
- `_notes/TASKS_TODO_PROMPTS.md` (design context for the loop that consumes this)

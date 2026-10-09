# Inline file prompts without slowing ex6

## Motivation

Add `_ex6/inline.py`: writing a text line ending in `;;;` launches a fresh coding agent with that prompt, source path, and line number, then removes the prompt from the file.

Primary constraint: ex6's TUI must remain fast. Inline prompting may take roughly 0.5 seconds to launch. No filesystem scan, Git command, queue polling, or inline-plugin callback may run in the TUI/render loop.

## Recommended design

Use `watchfiles` for event-driven wakeups and Git only after a candidate marker is found.

Do not poll the Git working tree. Polling would repeatedly spawn Git and contend for disk/CPU even when nothing changed. Do not add a per-frame hook to `_tui_loop`; it currently runs continuously with ~1 ms input waits.

Run one daemon watcher thread from `inline.setup(app)`:

1. `watchfiles.watch(root, watch_filter=..., debounce/step configured for about 500 ms)` sleeps in its native watcher while idle.
2. Coalesce each yielded batch to a set of changed regular-file paths.
3. Process paths serially on that same thread. A second worker/queue is unnecessary; delayed filesystem events can wait.
4. For each file, cheaply check current bytes for `;;;` and then exact line endings. If absent, do no Git work.
5. Only for candidate files, ask Git which current lines differ from `HEAD`.
6. Extract prompts, rewrite the file once, then add/invoke fresh contexts.

This keeps all plugin work off the UI thread and makes the idle cost effectively zero.

## GIL and responsiveness

A background Python thread is not enough by itself if it performs sustained Python CPU work. Keep its GIL ownership short and bounded:

- `watchfiles` waits through its Rust/native backend rather than busy-spinning in Python.
- Use watchfiles' batching instead of a Python sleep/poll loop.
- Run Git as a subprocess only for files containing a candidate marker. Subprocess execution is outside the Python GIL, and Python releases the GIL while waiting.
- Never scan the repository. Read only paths present in the debounced event batch.
- Coalesce duplicate paths before reading.
- Parse only compact `--unified=0` output.
- Rewrite each affected file once, even if it contains multiple prompt regions.
- Agent invocation is asynchronous already: `Context.invoke()` performs small setup and starts its own daemon thread.

The watcher briefly reacquires the GIL for filtering, bytes/line parsing, context creation, and `tui.add_context()`. These sections should contain no network waits and no unbounded repository-wide work. A 0.5-second debounce is preferred over launch latency optimizations.

Do not route events through a queue drained by every TUI frame. Existing code already adds temporary subagent contexts from worker threads. Under CPython, the short `contexts.append` performed by `tui.add_context()` is atomic under the GIL; inline should only append, never change `tui.current` or remove/reorder contexts.

## Watch filtering

Root is the project working directory, resolved once during setup. Require a Git work tree; disable inline with a debug message if root is not in one.

Compile the root `.gitignore` as requested:

```python
spec = PathSpec.from_lines(
    "gitwildmatch",
    (root / ".gitignore").read_text().splitlines(),
)

def watch_filter(change, path):
    relative_path = Path(path).resolve().relative_to(root).as_posix()
    return not spec.match_file(relative_path)
```

Compose this with watchfiles' `DefaultFilter`; supplying a custom filter otherwise replaces the default exclusions. This is important for `.git`, `__pycache__`, `.venv`, editor temporary files, and the Git index/lock churn caused by agents. `.git` must always be excluded even if absent from `.gitignore`.

If `.gitignore` is absent, use an empty spec. If root `.gitignore` changes, rebuild the spec for subsequent events. Scope is the root `.gitignore` represented by this explicit `PathSpec`; do not run `git check-ignore` from the event filter because that callback must remain cheap.

Dependencies:

- `watchfiles` is already installed.
- `pathspec` is not currently installed and must be added to the environment.

## Git line membership

Git is the source of truth for whether a tracked line is part of the prompt:

- Use `git diff HEAD --unified=0 --no-color --no-ext-diff -- <path>`.
- Comparing `HEAD` to the current working file includes staged and unstaged changes together.
- Parse each hunk's new range from `@@ -old +start,count @@`.
- New-range lines are the current file's changed/added line numbers. Pure deletions have count zero and contribute no lines.
- The `;;;` line itself must be in one of these ranges. An old marker elsewhere in the file must not launch when an unrelated edit is saved.
- Skip deleted paths, directories, symlinks, binary files, and text that cannot be decoded as UTF-8.

Use one status query for the debounced candidate paths to distinguish tracked from untracked, then diff only tracked candidates. Avoid repository-wide `git status` and avoid parsing filenames from patch headers; path-specific diffs keep spaces/unicode paths simple.

Untracked files have no Git baseline, so multiline membership cannot be inferred safely: treating every line as added could delete an entire newly created source file. Safe initial policy: support only the terminating marker line in untracked files. Tracked files receive full multiline behavior. If multiline prompts in untracked files become important, add an explicit delimiter later rather than guessing.

## Prompt extraction and deletion

For each tracked marker line:

1. Locate the maximal contiguous range of current line numbers belonging to the Git-added/changed ranges around that marker. This is the requested greedy up/down search.
2. Preserve lines exactly while locating and rewriting; use `splitlines(keepends=True)` so newline style and final-newline state survive.
3. Remove the final `;;;` from the terminating line in the text sent to the agent.
4. Send the collected lines in file order without adding unrelated unchanged context.
5. Capture the marker's original, one-based line number before deleting anything.
6. Delete the entire collected range from the file.

An unchanged line, including an unchanged blank line, terminates multiline collection. Adjacent unrelated changed lines are intentionally included because Git membership is the boundary requested by the feature.

If several disjoint changed ranges contain markers, collect all prompts from the original snapshot, delete ranges bottom-up, perform one file write, then launch one context per range. Treat a changed range as one prompt even if it accidentally contains multiple `;;;` endings; this avoids duplicate agents and overlapping deletions.

Before writing, verify the file still matches the bytes/stat snapshot used for extraction. If the editor saved again during Git/parsing work, do not write or launch; defer to the newer watch event. Delete successfully before launching so the plugin's own write event sees no marker and exits before running Git again.

## Agent creation

Each dispatch creates a fresh, visible `Context`; do not reuse or fork an active conversation.

- Reuse the normal coding-agent system prompt, tools, environment message, and AGENTS/CLAUDE message from `_ex6/agents.py`.
- Keep inline model, reasoning level, and provider as explicit constants in `inline.py` so there is no hidden model selection.
- Set context `cwd` to project root.
- Give it a unique readable name such as `inline_<basename>_<line>_<counter>`.
- Add it with `tui.add_context(ctx)` without changing current selection.
- Invoke exactly:

```text
(relative/path, line N)
prompt text
```

This makes source provenance explicit in model context and leaves every launched agent inspectable in selection mode.

## Lifecycle and failures

- Store watcher state/stop event in `app.plugin_data` so setup cannot start it twice.
- Use a daemon thread and a watchfiles stop event; optionally set it from `atexit` because ex6 has no plugin teardown hook.
- Catch watcher/plugin errors inside the thread and report them only through `app.debug_print()`. A malformed file or Git failure must not crash the TUI.
- Never use `print()` from the watcher.
- A failed read/diff/write does not launch an agent and leaves source untouched.

## Relevant files

- New `_ex6/inline.py`: watcher, filter, Git-range parser, extraction/deletion, context launch.
- Existing `_ex6/agents.py`: coding-agent messages/tools/model/provider configuration to reuse explicitly.
- Existing `ex6.py`: no planned changes; plugin setup already runs after TUI creation, `tui.add_context()` appends contexts, and `Context.invoke()` starts model work asynchronously.
- Root `.gitignore`: compiled by `PathSpec` for watch filtering.

## Verification

Keep tests small and helper-focused; use a temporary smoke script for integration, then delete it.

- Filter excludes `.gitignore` matches and watchfiles default exclusions.
- Idle watcher starts no Git process and does no periodic Python work.
- Rapid save events for one path coalesce and launch after about 0.5 seconds.
- Existing tracked single-line prompt is removed and dispatched with correct relative path/line.
- Tracked multiline prompt collects only contiguous changed current lines.
- Staged-only, unstaged-only, and mixed changes work through `diff HEAD`.
- Existing unchanged `;;;` is not dispatched after an unrelated edit.
- Untracked file deletes only marker line, not whole file.
- Plugin write causes a second event but no duplicate dispatch/Git diff.
- Concurrent editor save causes snapshot validation to skip stale deletion.
- Multiple disjoint prompt ranges rewrite once and launch once per range.
- UTF-8/newline preservation and paths containing spaces work.
- Stub model invocation during smoke test; verify TUI rendering/key handling continues while watcher is idle, during a burst of saves, and while Git is running.
- Inspect git diff/status after implementation and run existing tests.

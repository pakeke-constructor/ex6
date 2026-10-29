# toolbugs

## Motivation
User: "im more worried about the heredoc and CRLF issues you ran into."
During the perf work (see `perf.md`) the core tools silently corrupted ex6.py three times. Each time the tool reported success, so an agent can't notice unless it re-checks. For a harness whose whole pitch is explicit, trustworthy control, tools that silently mangle files are the worst kind of bug. Goal: find the root causes, fix them simply, and add a small repro so they stay fixed.

## Observed (Windows, Git Bash, Python 3.14, locale cp65001, git core.autocrlf=input)
1. Heredoc escapes lost one backslash level. The command was `python - <<'EOF' ... EOF` with a Python `'''...'''` string. The JSON tool arg contained `\\r\\n` (two chars each: backslash + r). The file ended up with real CR/LF characters (and a real 0x01 for `\\x01`), giving a SyntaxError. A quoted heredoc should not process backslashes, so somewhere between the tool arg and bash, one level of `\\` became `\`.
   - But `sed -i 's/^import math$/import math\\nimport itertools/'` (single quotes, not a heredoc) worked as expected. So the stripping isn't uniform; it depends on quoting context.
2. edit_file turned the whole file into CRLF (2580/2580 lines) after any edit. git then warned "CRLF will be replaced by LF".
3. Parallel tool calls on the same file misbehaved:
   - A batch of 3 parallel edit_file calls on ex6.py: 1 "Updated", 2 "search string not found (best 67%/43%)". Retrying the same searches one at a time succeeded.
   - The "Updated" edit from that batch (`import itertools`) later turned out to be missing from the file.
   - A read_file(lines=...) issued in parallel with an edit_file returned "ERROR: list index out of range"; the same read retried moments later worked.

## Hypotheses
- (1) `bash()` in `_ex6/tools.py` runs `[exe, "-c", command]` through `_windows_shell`. On Windows, subprocess joins argv with `list2cmdline`, and MSYS bash.exe re-parses the command line with its own (Cygwin) rules. Backslashes/quotes likely don't round-trip. Fix candidate: never put the script on the command line. Pass it via stdin (`bash -s`) or a temp `.sh` file, so there's no quoting layer at all.
  - First check whether the backslash is already missing in the `command` arg bash() receives (tool-arg decoding vs shell quoting): log `repr(command)`.
- (2) `edit_file` uses `open(p, "r")` / `open(p, "w")`: text mode, no `newline=`, no `encoding=`. On Windows, reading converts CRLF to `\n` and writing converts `\n` to CRLF, so every edit converts LF files to CRLF. It also only works with non-ASCII because this machine's locale happens to be UTF-8. Fix candidate: read/write with `encoding="utf-8", newline=""` and preserve the file's existing line ending (detect `\r\n` in the content; convert `search`/`replace` to match). Same fix for `write_file` (~L270; confirmed, it wrote this plan file as CRLF), `patch_file`, and `tasks.py` (`open(..., "w")` L57/L82).
- (3) `_get_file_lock` uses a plain dict `setdefault` + `if key not in locks` check, which is racy under parallel calls: two threads can create two different Locks for the same path. Also `read_file` probably doesn't take the lock, so it can see a half-written (truncated) file, since `open(p, "w")` truncates first. Fix candidates: create locks under a global lock (or `setdefault(key, threading.Lock())`), take the lock in read_file, and write atomically (temp file + `os.replace`). Also confirm how the harness runs parallel tool calls (threads? same ctx?) and whether the lost edit came from a stale read being written back.

## Steps
1. Repro script (temporary `smoke_test.py`, delete afterwards). Call `tools.bash` with commands containing `\\n`, `\\x01`, `"`, `'`, `$`, backslash before a quote, both in heredocs and in plain `-c`. Compare the output with the expected bytes. Log `repr(command)` at entry.
2. Fix bash (and check `powershell`, same `_windows_shell` path) so arbitrary command strings reach the shell byte-for-byte.
3. Fix line endings + encoding in edit_file/write_file/patch_file. The repro edits an LF file and a CRLF file and checks that both keep their endings and non-ASCII chars (`█─│`).
4. Fix locking: parallel threads doing edit_file x N + read_file on one file. All edits land, reads never see partial content.
5. Manual check in the TUI: an agent does parallel edits + heredoc python on a scratch file; git diff shows no line-ending churn.

## Files
- _ex6/tools.py: `_get_file_lock` (~L78), `write_file` (~L270), `patch_file` (~L360-400), `edit_file` (~L409), `read_file` (~L906), `bash` (~L1608), `powershell` (~L1638), `_windows_shell`
- _ex6/tasks.py: plan file writes (L57, L82)
- ex6.py: wherever parallel tool calls are dispatched (`call_tools`)

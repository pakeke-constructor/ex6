# batching: multi-read/multi-edit vs dropping the read-check
Status: parked (for discussion with user; don't code)

## Motivation
From invokelogs item 4: Opus makes 1129 multi-file/multi-range bash reads (`sed -n a,bp f; sed -n c,dp g`) and 611 python-heredoc edits (288 with 2+ replaces). Those reads/edits aren't tracked, which causes most of the 24 "Must read file before editing" errors.
User: "i wonder if this is too bloated; would rather keep the files simpler? The bash sed stuff; i wonder if there is a smart way to do that. Or alternatively, remove the read-check entirely - its 2026."

## How the check works today
- `ctx.mark_file_read(path, lines)` stores md5 of the file (+ line snapshots); `has_read_file` = hash still matches. (ex6.py ~1233)
- `write_file`: existing file must pass `_check_read`.
- `edit_file`: exact match needs NO read (match proves knowledge). Only whitespace/fuzzy tiers call `_check_read`.
- `edit_file_lines` uses line snapshots (separate; untouched here).
- So every failure = file read or changed via bash, or never read, then write_file/fuzzy edit.

## Note: bash reads are not the token problem
Opus's sed/grep results average 2.2k chars; its full-file read_file calls average 10.6k. Bash multi-range reads are the *efficient* pattern. The only cost is that they're untracked.

## Options
A. Batching params: `read_file(paths=[{path, lines}])`, `edit_file(edits=[...])`.
   + tracked reads, atomic multi-edit. - bigger schemas on every invoke, more code, Opus will likely keep using sed anyway (it's habit, Astra already uses ranges).
B. Parse bash commands for `cat f` / `sed -n a,bp f` and mark them read.
   + no schema change. - regex guessing on shell; misses heredocs, pipes, `sed -i`. Fragile.
C. Drop the read-check, keep a staleness check.
   - write_file: refuse only if this ctx has a hash for the file AND it changed since ("'x' changed since you last read/wrote it (bash, user or another agent). Re-read it."). Unknown files: allowed.
   - edit_file: drop `_check_read` from fuzzy tiers; instead report the matched span in the result when non-exact (`Updated x (fuzzy 86% match at lines 40-52)`), so a bad guess is visible.
   + ~-5 lines, keeps the useful part (other agents share the workspace; protects against clobbering concurrent edits). Kills all "never read" + "read via bash" failures.
   - "changed via bash after a tracked read" (~9 cases) still errors, but with a clear message. Blind write_file over an unseen file is allowed; git is the safety net.
D. Drop all checks. Simplest, but loses concurrent-edit protection; multiple agents in one workspace is normal here.

## Recommendation
C, and no batching params (skip A/B). Bash stays the multi-range read tool; maybe one line in the bash description: "`sed -n a,bp f` is fine for multi-range reads." Multi-edit heredocs are cheap (1.1M chars total) and work; leave them.

## Files
- `_ex6/tools.py`: `_check_read`, `write_file`, `edit_file`
- `ex6.py`: `Context.mark_file_read` / `has_read_file`

# edit_file_lines refactor plan

## Goal
Make edit_file_lines safe and deletion-friendly.
Agents can delete large blocks by line range without writing out the content.
The tool guarantees: if lines have shifted since last read, it errors.

Core value of edit_file_lines: Agents can delete large code blocks easily.
The agent should be reminded of this in the tool description.


## The core mechanism
Context stores `_line_snapshots: dict[str, list[str]]` — maps abs-path to full line list at time of read.

When agent calls read_file/read_headers/read_function, the snapshot is populated.
When agent calls edit_file_lines(file, start, end, content), we check:
  snapshot[start-1 : end] == current_file_lines[start-1 : end]
If mismatch: error "lines have shifted since last read, re-read the file."
If match: apply edit, update snapshot to new lines.

No explicit invalidation needed. After an edit:
- Lines above edited range: identical in new snapshot, still pass.
- Lines below edited range: shifted, will fail check with old numbers.
The content check IS the invalidation.

## KEY NUANCE (why the reset happened)
read_function and read_headers only return a SLICE of the file.
They must only snapshot the lines they actually returned, not the whole file.
Snapshotting the whole file from read_function would give false safety guarantees
for lines the agent never saw.

So:
- read_file: snapshots ALL lines (sees whole file)
- read_headers: snapshots only lines corresponding to the signatures it returned
- read_function: snapshots only the lines of the function body it returned

This means mark_file_read needs a way to merge/update a partial line snapshot,
not replace the whole thing.

## Data structure for partial snapshots
Store as dict[str, dict[int, str]]:  path -> {line_number: line_content}
(1-indexed to match the tool interface)

mark_file_read(path, lines=None, read_line_numbers=None):
  - updates _read_hashes as before
  - if lines + read_line_numbers provided: merges {n: lines[n-1] for n in read_line_numbers} into _line_snapshots[path]
  - read_line_numbers is a list[int] of 1-indexed line numbers the caller actually showed the agent

get_line_snapshot(path) -> dict[int, str]:
  - returns the known lines dict for that path

## Snapshot population per tool
read_file: line_dict = {i+1: line for i, line in enumerate(content.splitlines())}
read_function: line_dict = {start_line + i: line for i, line in enumerate(fn_text.splitlines())}
read_headers: line_dict = {line_no: sig_line} for each signature emitted (line_no already computed)

## edit_file_lines check
For each line number L in range(start, end+1):
  if L not in snapshot: error "line L not in any snapshot for this file, read it first"
  if snapshot[L] != current_lines[L-1]: error "line L has shifted since last read"

After successful edit: recompute snapshot for entire file (we have the new content anyway).
Or more simply: rebuild snapshot from new_lines fully, since we just wrote the file.
(Full rebuild is fine here — we just did a write, so we know all lines.)

## Changes needed

### ex6.py (Context class)
- Add field: _line_snapshots: dict[str, dict[int,str]] = field(default_factory=dict)
- Update mark_file_read(path, lines=None, read_line_numbers=None): merge line_dict into _line_snapshots[path]
- Add get_line_snapshot(path) -> dict
- Update clear(): reset _line_snapshots = {}
- Update fork(): copy _line_snapshots shallowly (dict of dicts, so copy one level deep)

### _ex6/tools.py

read_file:
  lines = content.splitlines()
  read_line_numbers = list(range(1, len(lines)+1))  # all lines
  ctx.mark_file_read(path, lines, read_line_numbers)

read_function:
  fn_lines = text.splitlines()
  read_line_numbers = list(range(start_line, start_line + len(fn_lines)))
  ctx.mark_file_read(file, all_file_lines, read_line_numbers)
  NOTE: needs to read whole file to pass `lines`, but only claims the function's line numbers.

read_headers:
  line_no is already computed per signature (source[:child.start_byte].count(b'\n') + 1)
  read_line_numbers = [line_no for each sig]
  ctx.mark_file_read(file, all_file_lines, read_line_numbers)
  NOTE: only claims the start-line of each definition.

edit_file_lines:
  - Remove old "WARNING: do not call twice" doc note
  - Add: "To delete, pass content='' "
  - Add snapshot check (per-line, as described above)
  - Fix deletion: new_lines[start-1:end] = [] when content == ""
  - After write: full snapshot rebuild from new_lines, ctx.mark_file_read(file, full_line_dict)

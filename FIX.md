# FIX: read_body + edit_file_lines snapshot mismatch

## The Bug
When an agent calls `read_body` then `edit_file_lines`, the edit gets rejected with
"Line N not in any snapshot" if the agent uses the boundary lines shown in read_body's output.

## Root Cause
`read_body` in `_ex6/tools.py` DISPLAYS lines `start_line-1` through `end_line+1` (±1 context),
but only SNAPSHOTS `start_line` through `end_line` (the tree-sitter node).

The agent sees the boundary lines in the output, naturally includes them in the edit range,
and `edit_file_lines`'s `assert_line` rejects them because they're not in the snapshot.

## Reproduction
```
read_body("test.lua", "hello")   # shows lines 9-12, snapshots only 10-12
edit_file_lines("test.lua", 9, 12, ...)  # FAILS: "Line 9 not in any snapshot"
```

Line 9 is `---@param w any` — a non-blank Lua annotation shown as context but not snapshotted.

## Fix 1: Snapshot the displayed range (main fix)
File: `_ex6/tools.py`, function `read_body`, inside the `if line_numbers:` branch.

Currently the snapshot is set BEFORE the display range is computed. Move it AFTER, and use `range(s, e+1)`:

```python
# CURRENT (broken):
read_line_numbers = list(range(start_line, end_line + 1))
ctx.mark_file_read(file, read_line_numbers)
if line_numbers:
    all_lines = source.decode().splitlines()
    s = max(start_line - 1, 1)
    e = min(end_line + 1, len(all_lines))
    ...

# FIXED:
if line_numbers:
    all_lines = source.decode().splitlines()
    s = max(start_line - 1, 1)
    e = min(end_line + 1, len(all_lines))
    read_line_numbers = list(range(s, e + 1))   # snapshot matches display
    ctx.mark_file_read(file, read_line_numbers)
    ...
read_line_numbers = list(range(start_line, end_line + 1))  # no-line-numbers path unchanged
ctx.mark_file_read(file, read_line_numbers)
return text
```

## Fix 2: Lua annotations in read_body
File: `_ex6/tools.py`, function `read_body`.

Lua `---@param` annotations are separate tree-sitter siblings ABOVE the function node.
The tree-sitter node starts at `function`, not at the annotations. So read_body doesn't
include them. Fix: walk `child.prev_sibling` (same pattern as `_signature_lua`) to extend
`start_byte` upward before computing `start_line`.

```python
start_byte = child.start_byte
sib = child.prev_sibling
while sib and sib.type == 'comment' and sib.text.decode().startswith('---'):
    start_byte = sib.start_byte
    sib = sib.prev_sibling
start_line = source[:start_byte].count(b'\n') + 1
end_line = source[:child.end_byte].count(b'\n') + 1
text = source[start_byte:child.end_byte].decode()
```

Note: `end_line` must be computed from `child.end_byte` directly, not from `start_line + len(lines) - 1`,
since `start_line` moved up but the node end didn't.

## Fix 3: Lua read_headers line numbers
File: `_ex6/tools.py`, functions `_read_headers_lua` and `read_headers`.

`_read_headers_lua` builds its own output text, then `read_headers` calls `_add_line_numbers(result)`
which numbers from 1 — giving WRONG line numbers. The non-Lua path embeds real `line_no` values.

Fix: make `_read_headers_lua` accept `line_numbers` param, compute real line numbers
(`line_no = source[:child.start_byte].count(b'\n') + 1`), embed them in output when requested,
and return `(result_str, sig_line_nos)`. Update the call site in `read_headers` to unpack
the tuple and use `sig_line_nos` for the snapshot instead of snapshotting all lines.

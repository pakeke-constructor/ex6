# outputfmt: condense GDScript read_headers / git_working_tree / search output
Status: parked (user wants to discuss; don't code yet)

## Motivation (user's words)
"look for common patterns that agents do that could be done with way less tokens. (Via smarter tools, or more condensed output.)"
On read_headers: "We will maybe want to replace the lua annotation stuff with typescript-style annotations, perhaps? do a pre-process step somehow? (Luau is in the training data so it's not too bad. My biggest concern is the complexity)"
This is item 7 of `.plans/invokelogs.md`, the one the user said matters most to get right.

## Data (analysis.db, Opus + Astra, 10-29..11-04)
read_headers results stay in context for ~13M carried chars. Lua (`src/g.lua`) is done (annotations folded into Luau-style sigs). GDScript:
- `src/mm.gd` (DUCK_DUCK_NUKE): 40k chars, 1056 lines. 535 are blank and 278 are the module `"""` docstring. Only 123 are `func`.
Simulated (`C:/tmp/inv_e/s/sim.py`): mm.gd 41k -> 16k (-61%), player.gd -17% (that file is mostly consts, which stay).

## 2. GDScript: stop emitting noise (line-based, no parser)
`_read_headers_gd` keeps every unindented line. Change it to:
- Skip blank lines. These are mostly empty lines inside function bodies, which aren't indented.
- Skip the contents of multi-line `"""` strings. Replace a module docstring with a single stub: `2: """ (docstring, 277 lines)"""`, so the agent knows it's there and can `read_file lines=[2,278]`.
- Skip plain `#` comments and keep `##` doc comments (Godot's doc syntax, which is the useful kind).
- Optional: fold `enum X {` ... `}` into `enum X { A, B, C }`.

Before:
```
1:
2: """
3:
4: # mm.gd -> mm stands for "multiplayer manager"
5: This class is intentionally a "god class". ALL NETWORKING ...
... (275 more docstring lines)
367: class RVChangeList:
369:
372:
375:
```
After:
```
2: """ (docstring, 277 lines)"""
291: class_name MM extends Node
...
367: class RVChangeList:
```
Complexity: about 15 lines changed in one function.

## 3. git_working_tree (Opus avg 13.8k, max 79k)
- Put `git diff --stat` first, then `diff -U1`. Unchanged context lines are 27% of the output; -U1 saves ~18%.
- Cap each file at ~3k chars with `... (N more lines; git diff -- <file>)`. The largest outputs are `_notes/*.md` and `ex6-revision.txt`, where the cap cuts 50–70%.
- Agents also run `git diff` through bash (1.6M chars). A bash/ENV_PROMPT hint to prefer `git_working_tree`, or `git diff -U1`, gets that too.
Complexity: about 20 lines. Split the diff on `diff --git` headers and truncate each chunk.

## 4. search (456 calls, 475k chars)
Print each file path once, with matches indented under it (-15%). Clip match lines at 300 chars (-5%).
```
src/a.py
  12: def foo(...)
  40: foo(1)
```
Complexity: about 10 lines. Track `last_file` in the loop.

## Open questions for the user
- GD: hide private `_func`s? 85 of mm.gd's 123 funcs are private. Leaning no.
- Should read_headers line numbers stay on by default? With one line per definition they cost little.

## Files
- `_ex6/tools.py`: `_read_headers_gd`, `read_headers`, `git_working_tree`, `search`
- Sample outputs: `C:/tmp/inv_e/*.txt` (logged), `C:/tmp/inv_e/s/*_after.txt` (simulated); `python scripts/invoke_insights.py` rebuilds analysis.db

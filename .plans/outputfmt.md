# outputfmt: condense read_headers / git_working_tree / search output
Status: parked (user wants to discuss; don't code yet)

## Motivation (user's words)
"look for common patterns that agents do that could be done with way less tokens. (Via smarter tools, or more condensed output.)"
On read_headers: "We will maybe want to replace the lua annotation stuff with typescript-style annotations, perhaps? do a pre-process step somehow? (Luau is in the training data so it's not too bad. My biggest concern is the complexity)"
This is item 7 of `.plans/invokelogs.md`, the one the user said matters most to get right.

## Data (analysis.db, Opus + Astra, 10-29..11-04)
read_headers results stay in context for ~13M carried chars. Two files account for the averages:
- `src/g.lua` (skullx11): 42k chars. 221 `---@param` lines, plus `@return`/`@field` lines, `local x = require(...)` lines, and a line number on every annotation line.
- `src/mm.gd` (DUCK_DUCK_NUKE): 40k chars, 1056 lines. 535 are blank and 278 are the module `"""` docstring. Only 123 are `func`.
Simulated the proposed formats on the logged outputs (`C:/tmp/inv_e/s/sim.py`): g.lua 42k -> 19k (-55%), abils.lua -47%, mm.gd 41k -> 16k (-61%), player.gd -17% (that file is mostly consts, which stay).

## 1. Lua: fold annotations into a Luau/TS-style signature (output-only)
Do this in the formatter only. No preprocessing step and no changes to source files: the annotation block (already collected by `_node_range_lua`) plus the signature line go through one pure function, `_lua_typed_sig(annotation_lines, sig) -> str`.

Before (current output):
```
85: --- Fires onBloodGained(amount). Clamped to [0, g.getMaxBlood()].
86: ---@param amount number
87: function g.addBlood(amount)

225: ---@param r number|objects.Color|string
226: ---@param green number?
227: ---@param b number?
228: ---@param a number?
229: ---@return objects.Color
230: function g.snapToPalette(r, green, b, a)

324: ---@param imageName string|love.Quad animated entities set ent.image to a Quad (a frame of a strip)
325: ---@return number w
326: ---@return number h
327: function g.getImageSize(imageName)

153: ---@class g.Terrain: g.TerrainDefinition
154: ---@field id string
155: ---@field blockSize integer footprint size in tiles. (16px image = 2x2 tiles.) Terrain placed in ...
156: ---@field columnQuads love.Quad[] [tileX % blockSize + 1] -> full-height column of the sprite, ...
157: local Terrain

4: local AutoAtlas = require("lib.AutoAtlas.AutoAtlas")
5: local imageCuts = require("src.modules.imageCuts")
```
After:
```
87: function g.addBlood(amount: number)  -- Fires onBloodGained(amount). Clamped to [0, g.getMaxBlood()].
230: function g.snapToPalette(r: number|objects.Color|string, green: number?, b: number?, a: number?): objects.Color
327: function g.getImageSize(imageName: string|love.Quad): (number, number)
153: class g.Terrain: g.TerrainDefinition { id: string, blockSize: integer, columnQuads: love.Quad[] }
```
(The `require` lines are dropped. One line per definition and no blank line between them, numbered at the `function` line.)

Rules:
- `@param name[?] type`: write `name: type` into the param list. `...` gets the same treatment.
- `@return type [name]`: one return gives `: T`, several give `: (A, B)` (Luau syntax).
- `@class X[: Base]` + `@field n t`: `class X: Base { n: t, ... }`.
- `---` description: keep only the first non-empty line, as a trailing `-- ...`.
- Drop param, return and field descriptions (open question below).
- Anything not understood (`@overload`, `@generic`, `@alias`, `@type`, a param name that isn't in the signature) stays verbatim above the signature. Unknown input is never lost silently; it just isn't condensed.
- Skip top-level `local x = require(...)`, so a file's import block disappears.
- Types can contain spaces and parens (`fun(e: g.Entity): boolean?`). Read the type with a small bracket-balanced scan, not `\S+`; the simulation broke on exactly this.

Complexity: about 50 lines in tools.py. One new pure function, small changes in `_read_headers_lua`/`_signature_lua`, and a handful of asserts for the pure function (g.lua cases above, including `fun(...)` and varargs). Tree-sitter is untouched. The risk is low: the worst case is today's output for one definition.

Why not preprocess source / convert to real TS annotations: it means a second representation to keep in sync, and agents editing the file would then see a different syntax from what's on disk. An output-only fold gives the same token win with no state.

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
- Lua: drop param/field descriptions (the biggest remaining chars), or keep them as trailing `-- desc` text? Today's plan drops them; `read_body` shows the full block.
- GD: hide private `_func`s? 85 of mm.gd's 123 funcs are private. Leaning no.
- Should read_headers line numbers stay on by default? With one line per definition they cost little.

## Files
- `_ex6/tools.py`: `_signature_lua`, `_node_range_lua`, `_read_headers_lua`, `_read_headers_gd`, `read_headers`, `git_working_tree`, `search`
- Sample outputs: `C:/tmp/inv_e/*.txt` (logged), `C:/tmp/inv_e/s/*_after.txt` (simulated); `python scripts/invoke_insights.py` rebuilds analysis.db

# perf

## Motivation
User: "ex6 feels noticably slower and clunkier than claude-code and codex, and i would like to address this."
Approach: measure first (profiler), then fix the biggest costs. Includes pragmatic Python-level wins: "avoiding `.put`, using `.puts` more, maybe using smarter python primitives".
Constraint: keep things simple. No clever abstractions; each step should be small and checked against the profiler.

## Current state
- `_ex6/profiler.py` exists, not tested yet. `/prof` toggles per-frame section timings and a HUD, and writes a report on stop. `/prof cpu [secs]` runs cProfile on the main thread. Reports go to `get_folder("profiler")`.
- An earlier render-cache attempt (per-message renderer cache + wrap-height cache) was reverted. Its lessons are in step 4.

## Microbenchmarks (200x50 buffer, headless, this machine)
- `flush` dirty-count loop on a frame where nothing changed: **2.6ms**. Comparing whole rows (`chars[y] != prev[y]`, C speed) instead: **0.045ms**. flush then scans every cell again with `any(...)` to pick dirty rows, so it's roughly 5ms per frame even when nothing changed.
- Writing 189 chars: `WrapWriter.put` per char 67us, `print_wrapped` 44us, `puts` 21us, slice assignment 2us. Writing cell by cell through `put` is 20-30x slower than slice writes.
- `buf.clear`: 0.1ms (fine).

## Done (2026-10-29), verified cell-for-cell vs HEAD on random inputs + plugin renderers
- Step 1: flush uses whole-row compares to find dirty rows (no per-cell loop, no 8% heuristic: only dirty rows are rewritten anyway). Runs grouped with itertools.groupby; `(fg, st, bg)` -> (prefix, suffix) escapes cached in `ScreenBuffer._style_cache`. Output bytes identical.
- Step 3: `ScreenBuffer.put_cells` (slice write, str or list of cells) under puts/fill/rect/rect_line/hline/clear_line/print_contained/print_wrapped (inlined slices); `WrapWriter.puts`. Callers moved: codeblock (tokens pre-merged per line by color), markdown (runs grouped by color/style), system prompt preview, render_tool_line, render_highlighted_line.
- Numbers (200x50, headless): frame+flush idle 8.4ms -> 0.45ms, all-dirty 18ms -> 9ms; puts 189ch 25us -> 5us; print_wrapped 47 -> 40us; 3000ch code block 1.4 -> 0.7ms; 40 md lines 0.9 -> 0.22ms.
- Step 2 skipped: on Windows blessed `kbhit` polls with `sleep(0.01)`, so each `inkey(timeout=0.001)` is ~10ms and the 4-call loop already idles at ~41ms/frame (~24fps). A single `inkey(1/60)` would render MORE often. Key latency is bounded by that 10ms poll.

## Plan (in order; measure before and after each step with /prof on the same long conversation)
0. Profiler: smoke-test `/prof` and `/prof cpu` in the real TUI. Get a baseline on a long conversation (lots of code blocks, markdown, tool rows) in work mode, both idle and streaming, plus selection mode. Record the numbers here.
1. `ScreenBuffer.flush` diff: find dirty rows with whole-row list compares across the 4 layers, and decide full vs diff by dirty-row count. Drop the per-cell Python loops. Maybe also cache `apply_style` per `(fg, st, bg)` (the blessed `color_rgb`/`getattr(term, attr)` calls per segment), but only if cProfile shows it matters.
2. Frame pacing / idle CPU: `_tui_loop` calls `inkey(timeout=0.001)` 4 times, so it renders almost constantly. Try a single `inkey(timeout=~1/60)` (check that blessed on Windows returns as soon as a key arrives, so latency stays the same). This also frees the GIL for LLM and tool threads (see the profiler's "stall" column). Skipping frames when nothing changed is out of scope unless this isn't enough.
3. Bulk-write primitives (the "avoid `.put`" step):
   - `puts`: clip once, then slice-assign all 4 layers (`chars[y][a:b] = text[i:j]`, `styles[y][a:b] = [style] * n`, ...).
   - `print_wrapped`: work out the wrapped segments per row, then write each with one `puts`. Output must match the current version cell for cell.
   - `WrapWriter.puts(text, ...)`: write runs that wrap across rows. Callers emit runs instead of single chars: codeblock tokens (`z_highlight_codeblock.py`), markdown cells grouped by `(color, style)` (`z_highlight_markdown.py`), the system-prompt preview (`z_render_system_prompts.py`), and `render_tool_line`'s `_put`.
   - Slice-based `hline`, `fill`, `fill_bg`, `rect_line`, `clear_line`.
   - Hot-loop basics: bind attributes to local names, `str.join`, no per-char function calls.
4. Work-mode render caching + off-screen culling. Only if the profiler still shows `render_work_mode` growing with conversation length after step 3. Lessons from the reverted attempt:
   - Cache the output of the `output_renderer` hooks per settled message. Key on text + role + theme identity, and store the msg and check `cached is msg`, because `id()` can be reused. Never cache the streaming message.
   - Off-screen lines: keep a `(line, w) -> height` cache and skip drawing anything above or below the view, including callable lines. Only store heights for settled messages, because streaming lines are rebuilt every frame and would flood the cache.
   - This changes the renderer contract: output may only depend on msg text, role, overview and theme. Document it in the architecture notes.
5. Selection mode and other panels: profile, then apply the step 3 primitives wherever they're hot.

## Verification
- Temporary `smoke_test.py`: compare the old vs new `puts`/`print_wrapped`/`WrapWriter` buffers cell for cell on random inputs (words longer than w, trailing spaces, negative x, y out of bounds, `\r\n`). Delete it afterwards.
- /prof report before and after each step; note the numbers here.

## Files
- ex6.py: `ScreenBuffer` (put/puts/print_wrapped/fill*/hline/rect_line/flush ~L1393-1600), `WrapWriter` (~L1615), `render_tool_line` (~L861), `render_work_mode` (~L2089), `_tui_loop` (~L2427)
- _ex6/profiler.py
- _ex6/z_highlight_codeblock.py, _ex6/z_highlight_markdown.py, _ex6/z_render_system_prompts.py

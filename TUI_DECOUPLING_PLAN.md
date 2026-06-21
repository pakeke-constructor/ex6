# TUI Decoupling Plan

Goal: make `ex6.py`'s runtime GENERIC — usable with no terminal UI. The TUI becomes an
opt-in object: construct `TUI()` and call `.run()` to get the interactive app. Without
instantiating it, you have a headless runtime (drive contexts from scripts, tests,
servers, whatever).

Everything stays in **one file, `ex6.py`**. "Decoupled" here means: the runtime classes
never reference TUI code, nothing starts the loop at import, and `TUI().run()` is the only
thing that boots the UI. Physical file location is irrelevant; the dependency direction is
what matters.

```python
import ex6
# headless: create contexts, invoke LLMs, run tools — no terminal needed
# interactive:
ex6.TUI().run()
```

---

## 1. Guiding principles

- Runtime code (Runtime, Context, Message, call_tools, tools, cost, plugins, commands)
  references NOTHING in the TUI half (ScreenBuffer, Terminal, Region, Theme, render_*,
  keys, modes).
- The `TUI` class may freely use runtime code. Never the reverse.
- Nothing runs the main loop at import time. The `__main__` block just does
  `_load_plugins()` then `TUI().run()`.
- Plugins keep working with minimal/zero churn. `import ex6` exposes the same names.
- No behavior change for the existing app. Pure restructure.

Layout within `ex6.py`: keep a clear top-to-bottom ordering — runtime section first, then
a visibly-marked TUI section (`# ===== TUI =====`). A reader should be able to see where
the runtime ends and the UI begins.

---

## 2. Coupling inventory (what must be severed)

Findings from reading `ex6.py` + `_ex6/`:

### Runtime-pure already (no change)
- `Message`, `ResponseChunk`, `LLMResult`
- `Context` core: `invoke()` loop + threading, `fork/clear/truncate`, file-read tracking,
  `get_tools`, `get_tool_schemas`
- `call_tools`, `tool_to_schema`, `_check_tool_args`, `_validate_tool_sig`, arg coercion
- `invoke_llm` override seam
- cost/budget, `_load_plugins`, command system (`@command`, `dispatch_command`, `_commands`)
- `overridable`/`override`, `after_tool_calls`/`_after_tool_calls`
- `debug_print`, `_debug_buffer`, `_StdoutSink`, `_thread_excepthook`, `get_folder`,
  `get_token_estimate`, `StrictDataDict`

### Coupling points (the work)

1. **`AppState` global `state`** (line 360) mixes runtime registry (`contexts`, `current`)
   with TUI state (`mode`, `_prev_mode`, `term`, `theme`).
   Plugins reference `ex6.state.contexts`, `.current`, `.theme`, `.mode` heavily
   (commands.py, tools.py, themes.py, agents.py, many `z_*`).

2. **`Context.__post_init__`** (722-725): registers self into `state.contexts` (runtime-ok)
   AND eagerly constructs an `InputBox` (TUI object).

3. **Context TUI-only fields**: `ui_stack`, `_input_box`, `_scroll_up`, `_prev_height`,
   `_tool_renderers`. Runtime only touches `_tool_renderers`, and that touch is itself a
   TUI concern (#4).

4. **`call_tools` -> `set_tool_renderer` -> `_default_tool_render`** (638, 597): runtime
   builds a closure that draws to a `ScreenBuffer`. Runtime never calls it; only the TUI
   does. But it makes `call_tools` transitively reference render code.

5. **Pure-TUI globals/functions**: `enter_scroll_mode`, `push_ui_panel`/`pop_ui_panel`/
   `_ui_panel_stack`, `output_renderer`, `Theme`, and the `_real_stdout/_real_stderr`
   fullscreen swap logic.

6. **The whole UI half**: `ScreenBuffer`, `WrapWriter`, `Region`, `InputPass`, `InputBox`,
   `make_input`, all `render_*`, the `__main__` main loop.

---

## 3. `AppState` -> `Runtime`

Rename `AppState` to `Runtime`. Holds ONLY runtime state:
```python
@dataclass
class Runtime:
    contexts: dict[str, Context] = field(default_factory=dict)
    current: Optional[Context] = None
```
Keep the global var named `state = Runtime()` so `ex6.state.contexts` / `ex6.state.current`
keep working unchanged in plugins.

`mode`, `_prev_mode`, `term`, `theme` move OFF Runtime, ONTO the `TUI` instance (§5).

---

## 4. `theme` / `mode` back-compat

~40 plugin refs to `ex6.state.theme` (tools.py, themes.py, all `z_*highlight*`,
render_system_prompts, checkpoints) + a few to `ex6.state.mode`. All inside rendering code
that only runs while a TUI is active.

Decision: `theme`/`mode` live on the `TUI` instance. Add a module-level `_active_tui` ref
(set in `TUI.__init__`), and give `Runtime` `theme`/`mode` **properties** that delegate to
`_active_tui`. So `ex6.state.theme` keeps resolving — zero plugin edits. Headless with no
TUI: property raises a clear error (these paths only execute under a running UI anyway).

`Theme` class definition moves into the TUI section of the file.

Since it's one file and `TUI()` is constructed in `__main__` BEFORE `_load_plugins()`, the
module-top-level `ex6.state.theme` reads in `z_highlight_*` resolve fine. (Confirm
`TUI.__init__` needs no plugins — it shouldn't.)

---

## 5. The `TUI` class

Lives in the TUI section of `ex6.py`:
```python
class TUI:
    def __init__(self, theme: Theme = None):
        global _active_tui
        self.term = Terminal()
        self.theme = theme or Theme()
        self.mode = "selection"
        self._prev_mode = "selection"
        self._ui_panel_stack = []
        self._sel_input_open = False
        self._sel_input_box = make_input(self._sel_on_submit)
        _active_tui = self

    def run(self):
        # the entire current __main__ while-loop body
        ...
```

Absorbed into `TUI` (as methods/attrs):
- the `__main__` while-loop -> `TUI.run()`
- `mode`, `_prev_mode`, `term`, `theme` -> instance attrs (no globals)
- `_ui_panel_stack`, `push_ui_panel`, `pop_ui_panel` -> instance
- `enter_scroll_mode` -> method
- the `_sel_*` selection-input state + `sel_on_submit`

Still module-level (the TUI section), because they're registries written at plugin-load
time, before any `TUI` exists:
- `output_renderer` decorator + its renderer list
- `_render_chunks` machinery

Plugin-facing forwarders kept at `ex6.` namespace so plugins don't change:
- `ex6.push_ui_panel(fn)` -> `_active_tui.push_ui_panel(fn)`
- `ex6.enter_scroll_mode()` -> `_active_tui.enter_scroll_mode()`
- `ex6.output_renderer` -> stays a module-level decorator (unchanged location/behavior)
- `ex6.state.theme` / `.mode` -> property shim (§4)

---

## 6. Severing the tool-renderer seam (#4)

Make `call_tools` emit data, let the TUI attach the draw closure.
- `call_tools` stops calling `_default_tool_render`.
- Add a runtime hook fired when a tool starts, e.g.
  `_on_tool_started(ctx, tool_call_id, name, args, thread, result)` (a small list of
  listeners, like `_after_tool_calls`).
- `TUI` registers a listener that does
  `ctx.set_tool_renderer(id, _default_tool_render(...))`.
- `render_tool_line` / `_default_tool_render` move to the TUI section.
- `Context.set_tool_renderer` / `_tool_renderers` stay on Context, unchanged. Headless:
  no listener registered -> no renderers created -> harmless.

(Fallback if the hook feels heavy: runtime stores a plain-data record
`{thread, result, name, args}` in `_tool_renderers`; TUI builds the draw fn at render
time. Prefer the hook.)

---

## 7. Step-by-step execution order

Small increments; app must still launch after each.

1. **Rename `AppState` -> `Runtime`**, strip `mode/_prev_mode/term/theme` fields.
   Temporarily leave them as no-op stubs if needed so the loop still runs. Verify launch.

2. **Add a `# ===== TUI =====` divider**; conceptually nothing below it may be referenced
   above it. Move `Theme` below the divider. Verify launch.

3. **Sever tool-renderer seam (§6)**: add `_on_tool_started` hook, move
   `_default_tool_render`/`render_tool_line` below the divider, TUI registers the listener
   (temporarily register it from `__main__` until `TUI` exists). Verify tool lines render
   (running/ok/error).

4. **Introduce `TUI` class**: move the `__main__` while-loop into `TUI.run()`, move
   `push_ui_panel`/`pop`/`_ui_panel_stack`/`enter_scroll_mode`, the `_sel_*` state, and the
   stdout-swap logic onto it. Add `ex6.push_ui_panel`/`enter_scroll_mode` forwarders.
   Verify full interactive app (selection, work, scroll, panels, commands).

5. **Move `theme`/`mode` onto `TUI`** + `_active_tui` bridge + `Runtime.theme`/`.mode`
   property shims. Remove the temporary stubs from step 1.
   Set `__main__` order to: `tui = TUI(); _load_plugins(); tui.run()` so top-level
   `ex6.state.theme` reads in `z_*` resolve. Verify highlighters, themes cmd, diff colors.

6. **Lazy `InputBox` in `Context`** (§2/§3): `_input_box` defaults `None`; TUI creates it on
   demand. `Context.__post_init__` no longer builds an InputBox. Keep
   `ui_stack`/`_scroll_up`/`_prev_height` fields (plugins use `ctx.push_ui`) but ensure no
   runtime logic depends on them. Verify `ctx.push_ui` (loading_cube, web_tools) works.

7. **Headless smoke test**: script that imports `ex6`, does NOT construct `TUI`, creates a
   `Context`, stubs `invoke_llm`, calls `ctx.invoke("hi")`, asserts a tool runs and a
   message appends — no Terminal touched. Acceptance test for the whole effort.

8. **Run `_ex6/_test.py`** + manual smoke of the real app.

---

## 8. Acceptance criteria

- `import ex6` with no `TUI()` constructed touches no terminal and starts no loop.
- Headless script (step 7) drives a full invoke->tool->loop cycle with no `TUI`.
- `ex6.TUI().run()` reproduces current app behavior exactly.
- All existing plugins load and function unchanged (forwarders + property shims).
- No runtime-section code references TUI-section symbols (ScreenBuffer/Terminal/Region/
  Theme/render_*).

---

## 9. Risks / watch-items

- **`ex6.state.theme` read at plugin import time**: `z_highlight_*`, `z_highlight_markdown`
  build color tables at module top-level. So `__main__` must construct `TUI()` BEFORE
  `_load_plugins()`. Confirm `TUI.__init__` needs no plugins.
- **`Context.fork`** already sets `_input_box=None` (line 856) — keep lazy creation
  consistent with that.
- **LLM/tool threads**: invoke loop uses `self` only; `call_tools` uses `ctx` only — no
  `state`/render access on worker threads. Keep this true after adding `_on_tool_started`
  (the hook only stores a closure; the closure runs later on the render thread).
- **stdout swap**: `_real_stdout`/`_real_stderr` captured at import (lines 24-25) stay in
  the runtime section as plain captured handles; the swap *logic* moves onto `TUI`.
- **`_fatal_error` / `_thread_excepthook`**: used by the loop. Loop moves to `TUI.run()`;
  the excepthook + global stay module-level. Fine.

# TUI Decoupling Plan

Goal: make `ex6.py`'s runtime GENERIC — usable with no terminal UI. The TUI becomes
opt-in: `ex6.run_tui()` boots the interactive app. Without calling it, you have a headless
runtime (drive contexts from scripts, tests, servers, whatever).

Everything stays in **one file, `ex6.py`**. "Decoupled" here means: the runtime classes
never reference TUI code, nothing starts the loop at import, and `ex6.run_tui()` is the
only thing that boots the UI. Physical file location is irrelevant; the dependency
direction is what matters.

```python
import ex6
# headless: create contexts, invoke LLMs, run tools — no terminal needed
# interactive:
ex6.run_tui()
```


---

## 1. Guiding principles

- Runtime code (Runtime, Context, Message, call_tools, tools, cost, plugins, commands)
  references NOTHING in the TUI half (ScreenBuffer, Terminal, Region, Theme, render_*,
  keys, modes).
- The `TUI` class may freely use runtime code. Never the reverse.
- Nothing runs the main loop at import time. The `__main__` block does
  `_load_plugins()` then `ex6.run_tui()`.
- Plugins migrate to the new accessors (`ex6.get_theme()`, `ctx.get_input_box()`).
  Breaking backwards compat is acceptable; keep the new shapes simple.
- No behavior change for the existing app's UX. Restructure + a few API renames.

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

Rename `AppState` to `Runtime`. Holds ONLY runtime state — keep it DEAD SIMPLE, no
properties, no shims, no magic:
```python
@dataclass
class Runtime:
    contexts: dict[str, Context] = field(default_factory=dict)
    current: Optional[Context] = None
```
Global var stays `state = Runtime()` so `ex6.state.contexts` / `ex6.state.current` keep
working in plugins.

`mode`, `_prev_mode`, `term`, `theme` move OFF Runtime, ONTO the `TUI` instance (§5).
It's FINE to break backwards compatibility here — do NOT add `@property` delegators on
Runtime. Plugins that read `ex6.state.theme` / `.mode` get migrated to the new accessors
(§4) instead.

---

## 4. `theme` access

~40 plugin refs to `ex6.state.theme` + a few to `ex6.state.mode`, all inside rendering code
that only runs while a TUI is active.

Decision: theme is owned by the `TUI` but exposed through plain module-level functions:
- `ex6.get_theme()` -> returns the active `Theme`
- `ex6.set_theme(th)` -> sets the active `Theme`

Theme is ALWAYS active (a default `Theme()` exists even when no TUI runs; sometimes unused,
that's fine). No `@property` shims. Migrate plugin `ex6.state.theme` reads to
`ex6.get_theme()`.

`mode` likewise lives on the `TUI` instance; plugins needing it go through the TUI (§5).

`Theme` class definition moves into the TUI section of the file.

---

## 5. The `TUI` class + module-level boot

`TUI` is a plain class. It is NOT instantiated directly by plugins or `__main__`. Two
module-level functions own its lifecycle:

- `ex6.run_tui()` — constructs the single `TUI` (once) and runs its blocking main loop.
  Asserts it isn't already running. `__main__` calls this.
- `ex6.get_tui()` — returns the live `TUI`, or `None` if headless / not yet started.
  Cheap, never blocks. Plugins use this to reach the active UI.

No `get_singleton()`, no raising `__init__`. Keeping "boot the app (blocking)" and "grab
the live TUI (cheap)" as separate functions avoids a get-or-start helper that sometimes
returns instantly and sometimes blocks forever.

```python
_tui = None  # the live TUI, or None when headless

def run_tui():
    global _tui
    assert _tui is None, "TUI already running"
    _tui = TUI()
    _tui.run()

def get_tui():
    return _tui  # None when headless

class TUI:
    def __init__(self):
        self.term = Terminal()
        self.mode = "selection"
        self._prev_mode = "selection"
        self._ui_panel_stack = []
        self._sel_input_open = False
        self._sel_input_box = make_input(self._sel_on_submit)

    def run(self):
        # the entire current __main__ while-loop body
        ...
```

Absorbed into `TUI` (as methods/attrs):
- the `__main__` while-loop -> `TUI.run()`
- `mode`, `_prev_mode`, `term` -> instance attrs (no globals)
- `_ui_panel_stack`, `push_ui_panel`, `pop_ui_panel` -> instance
- `enter_scroll_mode` -> method
- the `_sel_*` selection-input state + `sel_on_submit`

Theme is NOT a TUI instance attr; it lives behind `ex6.get_theme()` / `ex6.set_theme()`
(§4) and is always active (independent of the TUI), so plugin theme reads at import time
work even while `get_tui()` is still `None`.

Still module-level (the TUI section), because they're registries written at plugin-load
time, before any `TUI` exists:
- `output_renderer` decorator + its renderer list
- `_render_chunks` machinery

Plugin-facing forwarders kept at `ex6.` namespace so plugins don't change:
- `ex6.push_ui_panel(fn)` / `ex6.pop_ui_panel()` -> forward to `get_tui()`
- `ex6.enter_scroll_mode()` -> forward to `get_tui()`
- `ex6.output_renderer` -> stays a module-level decorator (unchanged location/behavior)


---

## 6. Severing the tool-renderer seam (#4)

Decision: keep runtime/tool state minimal, no new renderer model.

- `Context._tool_renderers` remains plugin override map (`tool_call_id -> callable`).
  Runtime does not write default draw closures into it.
- Add `Context._active_tools: dict[str, threading.Thread]` as in-flight tool tracker.
- `call_tools` updates only runtime state:
  - before `t.start()`: register `ctx._active_tools[tc_id] = t`
  - always cleanup in `finally`: `pop` every started id
  - still append normal `Message(role="tool", tool_call_id=...)` outputs as today
- TUI computes default tool line on render, per assistant `msg.tool_calls`:
  - if custom renderer exists in `_tool_renderers`, use it
  - else if `tool_call_id` in `_active_tools` and thread alive -> `running`
  - else inspect corresponding tool message:
    - content starts `ERROR:` -> `error`
    - otherwise -> `ok`
  - draw via `render_tool_line(...)`

This keeps one tiny in-flight source (`_active_tools`) and completed truth in messages.
No hook system, no extra view-model dicts, no runtime->TUI closure dependency.


---

## 7. Step-by-step execution order

Small increments; app must still launch after each.

1. **Rename `AppState` -> `Runtime`**, strip `mode/_prev_mode/term/theme` fields down to
   just `contexts`/`current`. No properties, no shims. Verify launch.

2. **Add a `# ===== TUI =====` divider**; conceptually nothing below it may be referenced
   above it. Move `Theme` below the divider. Add `get_theme()`/`set_theme()` (theme always
   active). Verify launch.

3. **Sever tool-renderer seam (§6)**: remove `_default_tool_render` from runtime; keep
   `render_tool_line` in TUI section. Add `Context._active_tools`, wire registration/cleanup
   in `call_tools`, and make work-mode render derive default tool rows from
   `_active_tools` + tool result messages. Verify running/ok/error rows.

4. **Introduce `TUI` + `run_tui()`/`get_tui()`**: move the `__main__` while-loop into
   `TUI.run()`, move `push_ui_panel`/`pop`/`_ui_panel_stack`/`enter_scroll_mode`, the
   `_sel_*` state, and the stdout-swap logic onto it. Add module-level `_tui`, `run_tui()`,
   `get_tui()`. Repoint `ex6.push_ui_panel`/`enter_scroll_mode` forwarders to `get_tui()`.
   Verify full interactive app.

5. **Migrate plugin theme/mode access**: change `ex6.state.theme` reads to `ex6.get_theme()`
   across plugins. `__main__` order: `_load_plugins(); ex6.run_tui()`.
   Verify highlighters, themes cmd, diff colors.

6. **Lazy `InputBox` via `Context.get_input_box()`** (§2/§3): add `Context.get_input_box()`
   that lazily builds the box; `_input_box` defaults `None`. `Context.__post_init__` no
   longer builds an InputBox. The TUI calls `ctx.get_input_box()` instead of a static
   helper. Keep `ui_stack`/`_scroll_up`/`_prev_height` fields; ensure no runtime logic
   depends on them. Verify `ctx.push_ui` works.

7. **`blessed` import only in TUI mode**: move `from blessed import Terminal` out of the
   module top and into the TUI section / `TUI.__init__` so headless `import ex6` never
   imports blessed (if feasible). Verify.

8. **Headless smoke test**: script that imports `ex6`, does NOT construct `TUI`, creates a
   `Context`, stubs `invoke_llm`, calls `ctx.invoke("hi")`, asserts a tool runs and a
   message appends — no Terminal touched.

9. **Run `_ex6/_test.py`** + manual smoke of the real app.

---

## 8. Acceptance criteria
- `import ex6` with no `run_tui()` called touches no terminal, starts no loop, imports no
  blessed. `get_tui()` returns `None`.
- Headless script drives a full invoke->tool->loop cycle with no `TUI`.
- `ex6.run_tui()` reproduces current app behavior. Calling it twice asserts.
- Plugins migrated to `ex6.get_theme()` / `ctx.get_input_box()` load and function.
- No runtime-section code references TUI-section symbols (ScreenBuffer/Terminal/Region/
  Theme/render_*).

---

## 9. Risks / watch-items

- **theme read at plugin import time**: `z_highlight_*`, `z_highlight_markdown` build color
  tables at module top-level via `ex6.get_theme()`. Theme must be always-active (default
  exists pre-TUI) so these resolve regardless of TUI construction order.
- **`Context.fork`** already sets `_input_box=None` — keep lazy `get_input_box()` consistent.
- **LLM/tool threads**: invoke loop uses `self` only; `call_tools` uses `ctx` only — no
  `state`/render access on worker threads. Keep this true through renderer-seam rework.
- **`_active_tools` cleanup correctness**: must clear on all exits (`stop_early`, exception,
  normal completion). Use `try/finally` around tool execution/join phase.
- **stdout swap**: `_real_stdout`/`_real_stderr` captured at import stay in the runtime
  section as plain captured handles; the swap *logic* moves onto `TUI`.
- **`_fatal_error` / `_thread_excepthook`**: loop moves to `TUI.run()`; the excepthook +
  global stay module-level. Fine.

---

## 10. Revised design decisions (supersede earlier drafts)

These reflect a deliberate rethink. Earlier sections were edited to match; this list is the
canonical summary of WHAT changed and WHY.

- **No `on_tool_started` hook.** Rejected.

- **Tool rendering seam finalized:** runtime tracks only in-flight tools via
  `Context._active_tools`; completed status comes from existing tool messages.
  `_tool_renderers` is plugin-custom callable override map only.

- **Input box via `Context.get_input_box()`.** No static helper on `TUI`. The Context lazily
  builds and owns its input box through a plain method.

- **`blessed` imports only in TUI mode.** Move the `from blessed import Terminal` into the
  TUI section / `TUI.__init__` so a headless `import ex6` never pulls in blessed.

- **`set_tool_renderer` stays** as plugin override path. Runtime default tool rows do not
  use it.

- **NO `@property` magic. `Runtime` stays dumb.** Drop all the property delegators and the
  `_active_tui` bridge. `Runtime` holds only `contexts` + `current`. Breaking backwards
  compatibility is GOOD here — keep it simple and obvious.

- **Theme via `ex6.get_theme()` / `ex6.set_theme(th)`.** Theme is ALWAYS active (a default
  exists even with no TUI; sometimes unused, that's fine). Plugins migrate off
  `ex6.state.theme`.

- **TUI lifecycle via two module funcs, not a singleton classmethod.** `ex6.run_tui()`
  builds the single `TUI` and runs the blocking loop (asserts not already running).
  `ex6.get_tui()` returns the live `TUI` or `None` (cheap, never blocks). No
  `get_singleton()`, no raising `__init__` — boot and access stay separate so plugins can
  reach the UI without risking launching the loop.


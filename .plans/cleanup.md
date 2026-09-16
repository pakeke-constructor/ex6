# Remove global runtime state

## Motivation

`ex6.py` currently has a bunch of weird global stuff: `_commands`, `_output_renderers`, `_after_tool_calls`, overrides, theme, current TUI, context schemas, budget state, and diagnostics. Global state is messy: ownership and lifetime are unclear, unrelated systems are coupled through module state, tests cannot isolate separate runtimes, and restarting or embedding ex6 risks retaining state.

Goal is not to add layers or turn ex6 into a framework. Keep ex6 hyper-minimal, but make runtime state explicit, locally owned, and easy to reason about. There should be a clear answer to “whose commands, contexts, theme, and hooks are these?” without relying on hidden module state.

## Direction

Introduce an explicit `App` runtime object. Mutable application state belongs to an `App`, rather than module-level globals or a hidden singleton.

Keep structure small:

- `App`: plugin registries, contexts, current context, theme, overrides, diagnostics, active TUI
- `TUI`: terminal/render/input state plus a reference to its owning `App`
- `Context`: conversation-specific state plus a reference to its owning `App`
- `DailyBudget`: persisted usage state and lock

Module scope should contain only classes, functions, type aliases, and constants.

## Plugin API

Preserve normal declaration syntax such as `@ex6.command`. These module-level decorators become stateless markers: they annotate functions but do not register them into global containers. After importing a plugin, the loader inspects declarations defined by that module and registers marked functions into the target `App`.

Use the same declaration model for:

- `@ex6.command`
- `@ex6.output_renderer`
- `@ex6.after_tool_calls`
- `@ex6.overridable`
- `@ex6.override`

This keeps plugin code flat and familiar while allowing multiple apps to load the same declarations independently. `app.command(fn)` and similar methods may still exist for deliberate dynamic registration, but should not be required for ordinary plugins.

Every registered command takes `tui` as its first parameter. Dispatch supplies the invoking TUI and excludes that parameter from command argument parsing/help text:

```python
@ex6.command
def usage(tui):
    ...

@ex6.command
def clr(tui, name: Optional[str]):
    ...
```

Commands should use `tui.app` for app-wide state and `tui.current` for the active context, replacing calls such as `ex6.get_current()`, `ex6.get_context()`, `ex6.push_ui_panel()`, and `ex6.get_theme()`.

Plugins may additionally expose `setup(app)` for imperative initialization such as creating contexts or configuring state. Loader imports plugins, registers their marked declarations, then invokes each setup function once in deterministic order.

Provide supported read APIs where needed, such as command iteration and retrieving an implementation to wrap. Plugins should not access private dictionaries like `ex6._commands` or `ex6.OVERRIDES`.

Avoid compatibility wrappers backed by a global "current app"; that would preserve the original problem.

## TUI ownership

- Construct `TUI(app)` and retain the owning app as `tui.app`.
- UI-local operations remain on TUI: current selection, panel stack, mode, input, terminal, and rendering state.
- App-owned services are reached through `tui.app`: contexts, theme, budget, command registry, overrides, and diagnostics.
- Command dispatch is a TUI operation (or receives the TUI explicitly) so it can invoke `fn(tui, *parsed_args)` without global lookup.

## Context ownership

- Remove automatic registration through module globals in `Context.__post_init__`.
- Construct detached contexts, then register with `app.add_context(ctx)`; optionally provide `app.create_context(...)` as shorthand.
- `app.add_context` assigns the context's owning app.
- Keep one private owning reference such as `ctx._app`; expose it publicly only if plugins have a real need.
- Context methods that require runtime ownership should fail clearly while detached.
- Route invocation, after-tool hooks, cloning, and forking through the owning app. Forked contexts remain owned and registered by the same app.
- Do not inject separate registry/service references into `Context`; one app reference is simpler and models ownership accurately.
- Move schema lookup/loading to `App`, since schema registry is app state.

## Other plugin callback changes

Commands are the callback type that clearly needs `tui` because they are initiated by a particular UI and commonly access current context or panels.

Other callbacks should follow their natural owner rather than all receiving TUI:

- Output renderers already receive `ctx`; use `ctx._app`/a supported context accessor for theme or app services. No TUI parameter needed unless renderer API is deliberately made UI-specific.
- `after_tool_calls(ctx)` already has context and therefore its owning app. No change needed.
- LLM providers and tool functions already receive `ctx`. Replace global budget, diagnostics, context removal, and theme access through context's app where needed.
- Render overrides already receive `tui`; use `tui.app` instead of global accessors.
- Imperative import-time actions—creating contexts, selecting current context, loading saved theme, setting budget—move to `setup(app)`.
- Helper functions invoked only from commands should accept `tui` or the narrower object they need instead of consulting ex6 globals.

## Runtime/process state

- Capture stdout/stderr when app starts, not at module import.
- Give `_StdoutSink` an app/debug callback.
- Install exception hooks for app lifetime and restore them during teardown.
- Move debug buffer, fatal error, and key logging flags into `App`/`TUI`.
- Move daily cost globals and persistence methods into `DailyBudget`.

## Implementation order

1. Add `App` and `DailyBudget`, initially preserving behavior.
2. Give `TUI` and registered `Context` instances an owning `App`; move context collection, current context, theme, and schema registry into it.
3. Make module-level registration decorators stateless declarations; move command, renderer, and after-tool registries into `App`. Change command dispatch to pass `tui` as first argument and migrate all commands.
4. Move override registry into `App`; route core overridable calls through it while keeping declaration decorators stateless.
5. Update plugin loader to register marked declarations and invoke optional `setup(app)` functions. Convert imperative in-repo plugin initialization to setup functions.
6. Move diagnostics, stdout/stderr handling, and exception state into runtime lifetime.
7. Remove obsolete module globals and compatibility accessors.
8. Search repository for remaining direct accesses to old state.

## Validation

- Two `App` instances have isolated registries, contexts, themes, schemas, overrides, and diagnostics.
- Starting an app twice does not retain plugin/runtime state.
- Marked plugin declarations register once per app despite plugins importing each other.
- Optional plugin setup runs exactly once per app.
- Each TUI points at its owning app; dispatch passes that TUI to commands and does not expose `tui` in command arguments/help.
- Detached contexts do not discover global state; registered contexts use only their owning app.
- Plugin callbacks use `tui.app`, context ownership, or setup injection rather than module runtime accessors.
- Commands, renderers, hooks, overrides, context fork/load, and budget persistence still work.
- stdout/stderr and exception hooks are restored after normal exit and failure.
- No mutable module-level runtime containers remain.

# Context(app, ...) and explicit TUI contexts

Motivation: separate execution dependencies from UI visibility. A context needs an app for providers, tools, hooks, budget, and plugin state; creating or cloning it must not silently add it to the TUI. Keep ex6 explicit and usable without a terminal.

API:
- `ctx = Context(app, name, model, ...)` associates the context with its app immediately.
- `tui.add_context(ctx)` explicitly displays it; `tui.get_context(name)` and `tui.remove_context(ctx)` manage displayed contexts.
- TUI owns `contexts` and `current`. Removing a displayed context does not detach it from its app.
- Remove App context CRUD/factory methods and Context detached-state machinery; no compatibility wrappers.
- Keep schema lookup on App. Context construction registers named schemas independently of TUI visibility.
- Forking, cloning, and App.load_context return app-associated contexts without displaying them. Commands explicitly add forks.

Files / implementation:
- `ex6.py`: required Context.app constructor field; TUI-owned list/selection and CRUD; independent schema registration; no automatic display from cloning.
- `_ex6/agents.py`, `_ex6/_test.py`: construct with app, explicitly add visible agents, select through app.tui.
- `_ex6/commands.py`, `_ex6/_plugin_examples/_example_commands.py`: use TUI CRUD and explicitly display forks; temporary commit-message context remains headless.
- `_ex6/generation.py`, `_ex6/sigils.py`: construct app-associated, headless contexts; remove obsolete registration cleanup.
- `_ex6/tools.py`, `_ex6/web_tools.py`: explicitly show temporary subagents when a TUI exists, then remove them without detaching execution dependencies; support headless calls.

Verification:
- Compile all Python files; check diff and stale old API references.
- Test invocation without TUI, independent TUI lists/selection, removal without detachment, app identity checks, forks/clones/schema loading without automatic display.
- Smoke-test local agent setup and plugin paths without network calls.

Scope: this tree only. External project plugins must migrate construction and UI registration separately.

Completed in this tree. Verification passed: 5 unittest cases in `tests/test_context.py`, syntax checks for 22 files, mocked-provider plugin smoke checks with and without TUI, full plugin loading, and `git diff --check`.




# ex6: Explicit LLM tool.

I was sick of all the LLM tools bloating my context window, and adding stuff implicitly that I didn't neccessarily want.  

I wanted a simpler, more explicit tool that allows for effective, nitty-gritty context engineering and workflow engineering.

This program, `ex6`, serves as a thin, simple alternative to claude-code, which does a lot less, and is entirely explicit.  
A lot of the UX/UI wi


## ex6 core tenets:
- No hidden context. Everything in the context window is explicit.
- High degree of customization/control.
- No leaky abstractions; serves as a thin, simple layer.

## features:
- less than 2000 loc pure python. No bloat.
- total customizability via "plugins."


## Plugins:
"Plugins" are just python files.  
"Installation?" Nope- just copy paste them into your `_ex6/` folder.  
All python files inside `_ex6` are loaded automatically.

Dependencies include Pillow (`pip install Pillow`) for the `read_image` tool.

## Invoke logging (headless or TUI)
Logging is off by default. Configure it directly from Python; no TUI or plugin loading is required:

```python
from _ex6.provider_util import configure_invoke_logging

configure_invoke_logging(True, path="./logs/invokes.db")
# Run contexts/providers normally; their completed invokes are logged automatically.
configure_invoke_logging(False)  # stop logging
configure_invoke_logging(True)   # resume at the same path
```

Configuration is process-wide, shared by all contexts/providers. In a plugin, make the same calls from `setup(app)`. Changes apply to subsequent `log_invoke` calls; an already-started write may finish.

`path` accepts a filename as a string or `Path`. Relative paths are resolved when configured; parent directories and the DB are created lazily on the first enabled write. Omitting `path` (or passing `None`) retains the chosen location. Without a custom path, the default is `ex6.get_folder("provider_util") / "invokes2.db"`.

Custom providers can call `log_invoke(ctx, provider, messages, tools, result, text="")` from `_ex6.provider_util` directly: `ctx` is an `ex6.Context`, `result` is an `ex6.LLMResult`, and messages/tools are the provider-format payload sent. No TUI is needed. Existing blob deduplication, base64 redaction, WAL transactions, and automatic culling above 1 GB are unchanged.




## Debug logs

Use `ctx.app.debug_print("label", value)` (or `app.debug_print(...)`) to dump values without touching the TUI. Dicts, lists, tuples, and sets are pretty-printed; `sep` and `end` are supported. Each call gets a timestamp and is written immediately as UTF-8.

Logs live in `ex6.get_folder("logs")/<project>-<path-hash>/debug.log`, exposed as `app.debug_path`: `%APPDATA%/ex6/logs/` on Windows, `~/ex6/logs/` on Unix. Project identity is the resolved working directory when the App is created (not the Git root): its folder name plus the first 12 hex characters of SHA-256 of its absolute path, case-normalized on Windows. Same-named projects in different directories stay separate. Each session starts with the full startup directory in a header. `/dbg` shows a snapshot in the same full-screen panel as `/help`, with the log path at the top. Up/Down scroll; Esc closes. Reopen `/dbg` to refresh. Agents can use existing file-reading/search tools on this plain-text file; no special log tool is needed. Logs aren't injected into model context.

Rotation keeps only two files per project:
- `debug.log`: current output.
- `debug.log.1`: previous chunk, not necessarily the previous session.
- Before a write would exceed 5 MiB (5,242,880 bytes), a nonempty current log replaces `.1` and writing starts in a fresh file.
- Creating an App also rotates a nonempty current log. An empty current log leaves `.1` intact.
- Older `.1` content is discarded. Normally this retains at most 10 MiB per project; a single oversized entry is kept whole and can exceed the limit. There is no global cap or cleanup of inactive project folders.

For live viewing, follow the filename across rotation (e.g. `tail -F <app.debug_path>`), not just the open file. Logs stay outside projects; no gitignore entries are needed. Dumps may contain sensitive values; review before sharing.


# DISCLAIMER:
This tool is NOT meant to be used for non-technical folk.
If any of the below is true:
- You don't care about understanding the codebase
- You don't care about code-quality
- You just want a tool that works instantly
- You don't understand how LLMs actually work
Then this tool *probably* isn't for you




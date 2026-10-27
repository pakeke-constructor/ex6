# native_tools: explicit native (provider-hosted) tools

## Motivation
Websearch works, but the current design is "ugly, and entirely implicit with global state":
- mode comes from `ctx.data["web_search"]` / `EX6_WEB_SEARCH` env / `/websearch` command, default "search" (implicitly ON for every context)
- `_native_web_stats` lives in `data_volatile`, keyed on the last user message
- every provider imports helpers from `provider.py`
That breaks ex6's core idea: every tool is explicit and visible in the context.

User's idea: "Expose an api to enable native-tools. `tools = [NATIVE_WEBSEARCH]`, then when it's openai, it's openai native, anthropic is anth native."

## Design
Core (ex6.py), small:
- `@dataclass(frozen=True) class NativeTool: __name__: str` (named `__name__` so with_tools/without_tools/get_tools/save/load all work by name unchanged).
- `_validate_tool_sig`, `Message.__post_init__`, `with_tools`: skip NativeTool.
- `get_tool_schemas()`: skip NativeTool. Add `get_native_tools() -> set[str]`.
- `call_tools`: native calls never reach ex6 (provider resolves them). If one does, error as unknown tool. No approval hook: native tools bypass `approve_tool` (by design, note in docs).
- `LLMResult.native_calls: list[dict]` ({"name", "args"}) per invoke. Replaces `_native_web_stats`. No cross-turn state.

Shared vocabulary (`_ex6/tools.py`, next to the other tool defs):
- `NATIVE_WEB_SEARCH = ex6.NativeTool("web_search")`
- `NATIVE_WEB_FETCH = ex6.NativeTool("web_fetch")`

Agents opt in explicitly: `tools=READ + [NATIVE_WEB_SEARCH]`. Default is off.

Providers map names -> their own native config. Unsupported name -> raise (explicit, no silent drop).
- anthropic: web_search -> `WebSearch`, web_fetch -> `WebFetch` in `--tools`/`--allowedTools`. Process reuse compares the native set (like `_mcp_tools`). `self.native_calls` id set stays (process-local, fine).
- openai: web_search -> `{"type":"web_search","external_web_access": "web_fetch" in natives}`. web_fetch without web_search -> raise.
- openrouter: raise for now (or map later, see Q4).

Delete: `_web_search_mode`, `_native_web_stats`, `/websearch` command, `EX6_WEB_SEARCH`, `ctx.data["web_search"]`.
Keep: cot chunk `[WebSearch] {...}` for visibility; `_log_invoke` logs `result.native_calls`.

## Open questions
1. Two tools (WEB_SEARCH, WEB_FETCH) vs one parameterized `NativeTool("web_search", fetch=True)`? Two maps 1:1 to Anthropic; OpenAI "fetch" is really "live access", slightly lossy.
2. Unsupported provider: raise (proposed) or silently ignore (lets same agent def run on any provider)?
3. Keep `native_calls` on LLMResult, or drop stats entirely (cot chunk + log only)?
4. OpenRouter: map web_search to its `web` plugin / `:online` now, or later?
5. Constants in core `ex6.py` vs `_ex6/tools.py` (proposed: class in core, constants in tools.py)?

## Files
- ex6.py: NativeTool, _validate_tool_sig, Message.with_tools/__post_init__, Context.get_tool_schemas/get_native_tools, LLMResult
- _ex6/tools.py: constants
- _ex6/provider.py: remove helpers + command; log native_calls
- _ex6/provider_anthropic.py, _ex6/provider_openai.py: map natives
- _ex6/agents.py: opt-in where wanted
- _notes/websearch.md, _readme.md: update

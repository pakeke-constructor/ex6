## Anthropic usage bar in footer

**Goal:** Show "x% used / 5h, resets in Xyz" bottom-right for Anthropic contexts, same as OpenAI codex bar but orange-colored with correct reset window.

**Data source:** Claude CLI (`claude -p --output-format=stream-json`) emits `rate_limit_event` events containing:
- `rate_limit_info.utilization` (0.0-1.0)
- `rate_limit_info.resetsAt` (unix timestamp)
- `rate_limit_info.rateLimitType` ("five_hour" / "seven_day")
- `rate_limit_info.unifiedWindows.{five_hour,seven_day}.{utilization,resetsAt}`

**Implementation:**
1. `provider_anthropic.py` `_read_stdout`: intercept `rate_limit_event`, store in `app.plugin_data["anthropic:usage"]`
2. Make the core footer `@handleable`; provider `@handler` functions claim their own contexts.
3. Render the Anthropic bar from `provider_anthropic.py` (orange, show binding window's reset time).

**Files:**
- `ex6.py` - handleable/handler chain
- `_ex6/provider_anthropic.py` - capture and render Anthropic usage
- `_ex6/provider_openai.py` - render OpenAI usage

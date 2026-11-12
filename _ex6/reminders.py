import ex6
from ex6 import Context

import time
from typing import Optional


def context_usage_reminder(remind: Optional[int] = 100_000, require: Optional[int] = 150_000):
    """after_tool_calls hook factory: appends `<context_usage>` to the last tool result every 10k tokens; nags handoff() past `remind`, demands it every call past `require`.
    ctx.data["reminders:handoff_remind"] / ["reminders:handoff_require"] override the thresholds at runtime (None disables)."""
    def hook(ctx: Context):
        msgs = ctx.get_messages()
        if not msgs or msgs[-1].role != "tool": return
        rem = ctx.data.get("reminders:handoff_remind", remind)
        req = ctx.data.get("reminders:handoff_require", require)
        used = ctx.token_count()
        bucket = used // 10_000
        last = ctx.data_volatile.get("reminders:context_usage_bucket", 0)
        ctx.data_volatile["reminders:context_usage_bucket"] = bucket # tracks downward too, so dd/p re-arms it
        if bucket <= last and (req is None or used < req): return
        note = f"{round(used/1000)}k/{round(ctx.max_tokens/1000)}k tokens used."
        if req is not None and used >= req:
            note += " YOU MUST call handoff() now."
        elif rem is not None and used >= rem:
            note += " Consider finishing your current step, then calling handoff()."
        r = msgs[-1].content
        msgs[-1].content = ex6.ToolResult(f"{r.text}\n\n<context_usage>{note}</context_usage>", r.attachments)
    return hook


def time_reminder(soft_limit: int = 5 * 60, subagent_limit: int = 2 * 60, interval: int = 30):
    """after_tool_calls hook factory: once the current invoke() run passes its soft limit, appends `<time_reminder>` to the last tool result every `interval` seconds.
    Subagents (ctx.data["tools:subagent_depth"]) default to `subagent_limit`. ctx.data["reminders:soft_time_limit"] overrides both (None disables)."""
    def hook(ctx: Context):
        msgs = ctx.get_messages()
        if not msgs or msgs[-1].role != "tool": return
        default = subagent_limit if "tools:subagent_depth" in ctx.data else soft_limit
        limit = ctx.data.get("reminders:soft_time_limit", default)
        if limit is None: return
        now = time.time()
        elapsed = now - ctx.invoke_time_start
        if elapsed < limit: return
        last = ctx.data_volatile.get("reminders:time_reminder_at", 0)
        if last >= ctx.invoke_time_start and now - last < interval: return
        ctx.data_volatile["reminders:time_reminder_at"] = now
        note = f"Running for {int(elapsed // 60)}m{int(elapsed % 60):02d}s (soft limit {int(limit // 60)}m{int(limit % 60):02d}s). Wrap up: finish your current step and report back."
        r = msgs[-1].content
        msgs[-1].content = ex6.ToolResult(f"{r.text}\n\n<time_reminder>{note}</time_reminder>", r.attachments)
    return hook

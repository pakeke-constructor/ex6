import ex6
from ex6 import Context

import os
import time
import difflib
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
        note = f"Running for {int(elapsed // 60)}m{int(elapsed % 60):02d}s (soft limit {int(limit // 60)}m{int(limit % 60):02d}s). Be wary that the human might be waiting."
        r = msgs[-1].content
        msgs[-1].content = ex6.ToolResult(f"{r.text}\n\n<time_reminder>{note}</time_reminder>", r.attachments)
    return hook


def files_changed_reminder(max_files: int = 10):
    """after_tool_calls hook factory: appends `<files_changed>a.py (+3 -1), ...</files_changed>` to the last tool result when files this ctx read/wrote were changed by someone else.
    Paths and line counts only, never diffs; agent uses read_file(since_last_read=True) to see them. Each change is reported once."""
    def hook(ctx: Context):
        msgs = ctx.get_messages()
        if not msgs or msgs[-1].role != "tool": return
        seen = ctx.data_volatile.setdefault("reminders:files_changed_seen", {})
        items = []
        for p, old, new in ctx.changed_files():
            if seen.get(p) == hash(new): continue
            seen[p] = hash(new)
            try:
                rel = os.path.relpath(p, ctx.cwd or os.getcwd())
            except ValueError: # other drive
                rel = p
            if new is None:
                items.append(f"{rel} (deleted)")
                continue
            diff = difflib.unified_diff(old.decode(errors="replace").splitlines(), new.decode(errors="replace").splitlines(), lineterm="", n=0)
            plus = minus = 0
            for line in diff:
                if line.startswith("+") and not line.startswith("+++"): plus += 1
                elif line.startswith("-") and not line.startswith("---"): minus += 1
            items.append(f"{rel} (+{plus} -{minus})")
        if not items: return
        note = ", ".join(items[:max_files])
        if len(items) > max_files:
            note += f", ...and {len(items) - max_files} more"
        r = msgs[-1].content
        msgs[-1].content = ex6.ToolResult(f"{r.text}\n\n<files_changed>{note}</files_changed>", r.attachments)
    return hook

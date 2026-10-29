# Tool call stats: one JSONL line per tool call, monthly files in get_folder("toolstats"). /toolstats shows counts.
# Token counts are ex6.get_token_estimate() estimates, not provider-reported.

import json
import os
import threading
import time
from collections import Counter
from typing import Optional
import ex6

_lock = threading.Lock()
ARG_CHARS = 100  # string args truncated to this many chars


def _write(ctx, name, args, text):
    args = args or {}
    row = {
        "ts": time.time(),
        "cwd": ctx.cwd or os.getcwd(),
        "ctx": ctx.name,
        "model": ctx.model,
        "tool": name,
        "args": {k: v[:ARG_CHARS] if isinstance(v, str) else v for k, v in args.items()},
        "error": text.startswith("ERROR:"),
        "preview": " ".join(text[:200].split())[:80],
        "bytes": len(text.encode()),
        "args_tokens": ex6.get_token_estimate(json.dumps(args, default=str), ctx.model),
        "result_tokens": ex6.get_token_estimate(text, ctx.model),
    }
    path = ex6.get_folder("toolstats") / time.strftime("%Y-%m.jsonl")
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


@ex6.after_tool_calls
def log_tool_calls(ctx):
    msgs = ctx.get_messages()
    results = {}
    i = len(msgs) - 1
    while i >= 0 and msgs[i].role == "tool":
        results[msgs[i].tool_call_id] = msgs[i].content
        i -= 1
    if i < 0 or not msgs[i].tool_calls: return
    for tc in msgs[i].tool_calls:
        _write(ctx, tc["name"], tc.get("args"), ex6.tool_result_text(results.get(tc["id"], "")))


@ex6.after_llm_turn
def log_native_calls(ctx):
    if not ctx.llm_result: return
    for nc in ctx.llm_result.native_calls or ():
        _write(ctx, "native:" + nc["name"], nc.get("args"), str(nc.get("result") or ""))


@ex6.command
def toolstats(tui, month: Optional[str]):
    'Tool call counts and estimated tokens. Arg: YYYY-MM (default current month) or all.'
    from _ex6.commands import _text_panel
    folder = ex6.get_folder("toolstats")
    pattern = "*.jsonl" if month == "all" else f"{month or time.strftime('%Y-%m')}.jsonl"
    counts, args_tok, result_tok = Counter(), Counter(), Counter()
    for path in folder.glob(pattern):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            counts[row["tool"]] += 1
            args_tok[row["tool"]] += row.get("args_tokens", 0)
            result_tok[row["tool"]] += row.get("result_tokens", 0)
    lines = [f"{pattern}: {sum(counts.values())} calls", "", f"{'calls':>7}  {'args_tok':>9}  {'result_tok':>10}  tool"]
    lines += [f"{n:>7}  {args_tok[name]:>9}  {result_tok[name]:>10}  {name}" for name, n in counts.most_common()]
    _text_panel(tui, lines)

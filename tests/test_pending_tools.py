import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import ex6


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for tool state"
        time.sleep(0.001)


def call(name, id=None, **args):
    return {"name": name, "id": id or name, "args": args}


def context(tools):
    app = ex6.App()
    ctx = app.add_context(ex6.Context("test", "", messages=[
        ex6.Message("system", "", tools=tools),
    ]))
    return app, ctx


def render(app, ctx):
    rows = []
    tui = SimpleNamespace(app=app, current=ctx, show_cot=True)
    buf = ex6.ScreenBuffer(100, 40)
    with patch.object(ex6, "render_tool_line", side_effect=lambda *args: rows.append(args[4:8])):
        ex6.render_work_mode(tui, buf, ex6.InputPass([]), ex6.Region(0, 0, 100, 40))
    return rows, buf


def batch(action):
    release = threading.Event()
    entered = threading.Event()
    attachment = ex6.Attachment("image.png", "image/png")

    def slow(ctx):
        entered.set()
        assert release.wait(3)
        return "slow result"

    def empty(ctx, tool_call_id: str):
        assert tool_call_id == "empty"
        return ex6.ToolResult("", (attachment,))

    def broken(ctx):
        raise ValueError("broken")

    def large(ctx):
        return "x" * (ex6.MAX_TOOL_OUTPUT_CHARACTERS + 1)

    def number(ctx, value: int):
        assert value == 7
        return value

    app, ctx = context([slow, empty, broken, large, number])
    calls = [call("slow"), call("empty"), call("broken"), call("missing"),
             call("large"), call("number", value="7")]
    result = ex6.LLMResult(tool_calls=calls)
    hooks = []

    def llm(ctx):
        yield ex6.ResponseChunk("text", "working")
        yield result

    @ctx.after_llm_turn
    def after_turn(ctx):
        assert ctx.llm_result is result
        assert ctx.llm_current_output[0].content == "working"
        assert [m.role for m in ctx.get_messages()] == ["system", "user"]
        hooks.append("turn")

    @ctx.after_tool_calls
    def after_tools(ctx):
        assert ctx.pending_tools is None
        roles = [m.role for m in ctx.get_messages()]
        assert roles == (["system"] if action == "truncate" else ["system", "user", "assistant"] + ["tool"] * 6)
        hooks.append("tools")
        ctx.stop_early = True

    threads = []
    start = threading.Thread.start

    def capture(thread):
        threads.append(thread)
        start(thread)

    with patch.object(threading.Thread, "start", capture):
        ctx.invoke("go", llm)
        try:
            assert entered.wait(3)
            wait_for(lambda: ctx.pending_tools is not None and all(p.result is not None for p in ctx.pending_tools[1:]))
            pending = ctx.pending_tools
            assert pending[0].result is None
            assert pending[1].result == ex6.ToolResult("", (attachment,))
            assert pending[2].result.text == "ERROR: broken"
            assert pending[3].result.text == "ERROR: Unknown tool: missing"
            assert pending[4].result.text.startswith("ERROR: Tool output too large")
            assert pending[5].result.text == "7"
            assert [m.role for m in ctx.get_messages()] == ["system", "user"]
            clone = ctx.fork("clone")
            assert clone.pending_tools is None
            assert [m.role for m in clone.get_messages()] == ["system", "user"]
            rows, buf = render(app, ctx)
            assert len(rows) == 6
            assert [row[2] for row in rows] == ["running", "ok", "error", "error", "error", "ok"]
            assert rows[1][3] == ""
            assert sum("".join(row).count("working") for row in buf.chars) == 1
            assert not any("█" in row for row in buf.chars)
            if action == "stop":
                ctx.stop_early = True
            elif action == "clear":
                ctx.clear()
                assert ctx._clear_pending
            elif action == "truncate":
                ctx.truncate(1)
            if action != "success":
                assert ctx.pending_tools is pending
                assert ctx.is_running()
                try:
                    ctx.invoke("again", llm)
                except RuntimeError:
                    pass
                else:
                    assert False, "Invoke must reject draining context"
                assert not any(m.role == "assistant" for m in ctx.get_messages())
        finally:
            release.set()
            for thread in threads:
                thread.join(3)
                assert not thread.is_alive()

    assert not ctx.is_running()
    assert ctx.pending_tools is None
    assert hooks == (["turn", "tools"] if action in ("success", "truncate") else ["turn"])
    if action == "success":
        messages = ctx.get_messages()
        assert messages[2].content == "working"
        assert messages[2].tool_calls == calls
        assert [m.tool_call_id for m in messages[3:]] == [tc["id"] for tc in calls]
        assert [m.content for m in messages[3:]] == [p.result for p in pending]
        rows, _ = render(app, ctx)
        assert len(rows) == 6
        assert [row[2] for row in rows] == ["ok", "ok", "error", "error", "error", "ok"]
    else:
        assert [m.role for m in ctx.get_messages()] == (["system", "user"] if action == "stop" else ["system"])


def coordinator_failure():
    release = threading.Event()
    entered = threading.Event()
    errors = []

    def slow(ctx):
        entered.set()
        assert release.wait(3)
        return "done"

    _, ctx = context([slow])

    def run_batch():
        try:
            ex6.call_tools(ctx, ex6.LLMResult(tool_calls=[call("slow"), {}]))
        except KeyError as e:
            errors.append(e)

    thread = threading.Thread(target=run_batch)
    thread.start()
    try:
        assert entered.wait(3)
        assert ctx.pending_tools is not None
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert len(errors) == 1
    assert ctx.pending_tools is None
    assert [m.role for m in ctx.get_messages()] == ["system"]


def run():
    for action in ("success", "stop", "clear", "truncate"):
        batch(action)
    coordinator_failure()


run()

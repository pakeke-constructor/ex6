import threading
import time
import unittest

import ex6
from _ex6 import tools


class ScheduleTests(unittest.TestCase):
    def test_idle_callback_receives_args_and_runs_on_caller(self):
        ctx = ex6.Context("test", model="test")
        calls = []
        caller = threading.current_thread()
        ctx.schedule(lambda value, flag: calls.append((value, flag, threading.current_thread())), 42, flag=True)
        self.assertEqual(calls, [(42, True, caller)])

    def test_only_one_pending_callback_and_clear_cancels_it(self):
        ctx = ex6.Context("test", model="test", llm_is_running=True)
        calls = []
        ctx.schedule(calls.append, "first")
        with self.assertRaisesRegex(RuntimeError, "already scheduled"):
            ctx.schedule(calls.append, "second")
        self.assertEqual(ctx._scheduled[1], ("first",))
        self.assertFalse(ctx.stop_early)
        ctx.clear()
        self.assertIsNone(ctx._scheduled)
        self.assertEqual(calls, [])

    def test_callback_runs_after_stream_and_cleanup(self):
        streaming = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        calls = []
        model_threads = []

        def provider(ctx):
            model_threads.append(threading.current_thread())
            streaming.set()
            release.wait(5)
            yield ex6.ResponseChunk(type="text", content="Done")
            yield ex6.LLMResult()

        ctx = ex6.App().create_context("test", model="test", invoke_llm=provider)

        def callback(value, flag):
            calls.append((value, flag, ctx.is_running(), ctx.llm_suspended,
                          ctx.last_invoke_time_end, ctx._scheduled,
                          ctx.get_messages()[-1].content, threading.current_thread()))
            ctx.schedule(calls.append, "nested")
            finished.set()

        ctx.invoke("Prompt")
        try:
            self.assertTrue(streaming.wait(5))
            ctx.schedule(callback, 42, flag=True)
            self.assertEqual(calls, [])
        finally:
            release.set()
        self.assertTrue(finished.wait(5))
        self.assertEqual(calls[0][:4], (42, True, False, False))
        self.assertGreater(calls[0][4], 0)
        self.assertEqual(calls[0][5:7], (None, "Done"))
        self.assertIs(calls[0][7], model_threads[0])
        self.assertEqual(calls[1], "nested")

    def test_stopped_run_still_drains_tools_before_callback(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        states = []
        tool_threads = []

        def slow_tool(ctx: ex6.Context) -> str:
            tool_threads.append(threading.current_thread())
            started.set()
            release.wait(5)
            return "Done"

        def provider(ctx):
            yield ex6.LLMResult(tool_calls=[{"id": "slow", "name": "slow_tool", "args": {}}])

        ctx = ex6.App().create_context("test", model="test", invoke_llm=provider, messages=[
            ex6.Message(role="system", content="Instructions", tools=[slow_tool]),
        ])

        def callback():
            states.append((ctx.is_running(), dict(ctx._active_tools), tool_threads[0].is_alive()))
            finished.set()

        ctx.invoke("Prompt")
        try:
            self.assertTrue(started.wait(5))
            ctx.schedule(callback)
            ctx.stop_early = True
            self.assertFalse(finished.wait(0.15))
            self.assertTrue(ctx._active_tools["slow"].is_alive())
        finally:
            release.set()
        self.assertTrue(finished.wait(5))
        self.assertEqual(states, [(False, {}, False)])


class HandoffTests(unittest.TestCase):
    def test_handoff_waits_for_tools_then_starts_clean_run(self):
        started = threading.Event()
        release = threading.Event()
        scheduled = threading.Event()
        finished = threading.Event()
        requests = []
        model_threads = []
        tool_threads = []
        states = []

        def slow_tool(ctx: ex6.Context) -> str:
            tool_threads.append(threading.current_thread())
            started.set()
            release.wait(5)
            ctx.append_message(ex6.Message(role="user", content="Stale message"))
            ctx.data_volatile["stale"] = []
            return "Stale result"

        def request_handoff(ctx: ex6.Context, txt: str) -> str:
            started.wait(5)
            result = tools.handoff(ctx, txt)
            scheduled.set()
            return result

        def provider(ctx):
            requests.append([(m.role, m.content) for m in ctx.get_messages()])
            model_threads.append(threading.current_thread())
            if len(requests) == 1:
                yield ex6.LLMResult(tool_calls=[
                    {"id": "slow", "name": "slow_tool", "args": {}},
                    {"id": "handoff", "name": "request_handoff", "args": {"txt": "New prompt"}},
                ])
            else:
                states.append((dict(ctx._active_tools), dict(ctx._read_hashes),
                               dict(ctx._line_snapshots), dict(ctx.data_volatile),
                               tool_threads[0].is_alive()))
                yield ex6.ResponseChunk(type="text", content="Done")
                yield ex6.LLMResult()

        system = ex6.Message(role="system", content="Instructions", tools=[slow_tool, request_handoff, tools.handoff])
        ctx = ex6.App().create_context("test", model="test", messages=[system], invoke_llm=provider)
        ctx.after_llm_turn(lambda ctx: finished.set() if ctx.get_messages()[-1].content == "Done" else None)
        ctx._read_hashes["old.py"] = "hash"
        ctx._line_snapshots["old.py"] = {1: "old"}
        ctx.data_volatile["old"] = []
        ctx.invoke("Old prompt")
        try:
            self.assertTrue(scheduled.wait(5))
            self.assertEqual(len(requests), 1)
            self.assertTrue(ctx.is_running())
            self.assertFalse(ctx.stop_early)
            self.assertTrue(ctx._active_tools["slow"].is_alive())
        finally:
            release.set()
        self.assertTrue(finished.wait(5))
        deadline = time.monotonic() + 5
        while ctx.is_running() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(ctx.is_running())
        self.assertEqual(states, [({}, {}, {}, {}, False)])
        self.assertEqual(requests, [
            [("system", "Instructions"), ("user", "Old prompt")],
            [("system", "Instructions"), ("user", "New prompt")],
        ])
        self.assertIsNot(model_threads[0], model_threads[1])
        self.assertEqual([(m.role, m.content) for m in ctx.get_messages()], [
            ("system", "Instructions"), ("user", "New prompt"), ("assistant", "Done"),
        ])
        self.assertIs(ctx.get_messages()[0], system)
        self.assertIn("handoff", ctx.get_tools())
        schema = ex6.tool_to_schema("handoff", tools.handoff)
        self.assertEqual(schema["function"]["parameters"]["required"], ["txt"])


if __name__ == "__main__":
    unittest.main()

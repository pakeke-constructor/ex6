import threading
import unittest
from unittest.mock import patch

import ex6


class ContextTests(unittest.TestCase):
    def make_tui(self, app):
        with patch.object(ex6.TUI, "__post_init__"):
            tui = ex6.TUI(app)
        app.tui = tui
        return tui

    def test_construction_does_not_display_context(self):
        app = ex6.App()
        tui = self.make_tui(app)
        ctx = ex6.Context(app, "hidden", "test")
        self.assertIs(ctx.app, app)
        self.assertEqual(tui.contexts, [])
        self.assertIsNone(tui.current)
        self.assertFalse(hasattr(app, "contexts"))
        self.assertFalse(hasattr(app, "create_context"))
        self.assertFalse(hasattr(app, "add_context"))

    def test_tui_add_lookup_and_remove(self):
        app = ex6.App()
        tui = self.make_tui(app)
        first = ex6.Context(app, "first", "test")
        second = ex6.Context(app, "second", "test")
        self.assertIs(tui.add_context(first), first)
        tui.add_context(first)
        tui.add_context(second)
        self.assertEqual(tui.contexts, [first, second])
        self.assertIs(tui.current, first)
        self.assertIs(tui.get_context("second"), second)
        tui.remove_context(first)
        self.assertIs(tui.current, second)
        self.assertIs(first.app, app)
        self.assertIsNone(tui.get_context("first"))
        tui.remove_context(second)
        self.assertIsNone(tui.current)

    def test_tui_lists_are_independent_and_reject_other_apps(self):
        app = ex6.App()
        first = self.make_tui(app)
        second = self.make_tui(app)
        ctx = ex6.Context(app, "ctx", "test")
        first.add_context(ctx)
        self.assertEqual(second.contexts, [])
        self.assertIsNone(second.current)
        other = ex6.Context(ex6.App(), "other", "test")
        with self.assertRaises(RuntimeError):
            first.add_context(other)

    def test_fork_and_schema_load_do_not_display_contexts(self):
        app = ex6.App()
        tui = self.make_tui(app)
        source = ex6.Context(app, "source", "test", schema_id="test-schema",
                             messages=[ex6.Message("user", "hello")])
        self.assertIs(app.context_schemas["test-schema"], source)
        tui.add_context(source)
        source.data["key"] = "value"
        fork = source.fork("fork")
        clone = source.clone_with_context(source.dump_context())
        loaded = app.load_context(source.dump_context())
        for ctx in (fork, clone, loaded):
            self.assertIs(ctx.app, app)
            self.assertIsNot(ctx.get_messages()[0], source.get_messages()[0])
            self.assertEqual(ctx.get_messages()[0].content, "hello")
            self.assertEqual(ctx.data["key"], "value")
        self.assertEqual(tui.contexts, [source])
        tui.add_context(fork)
        self.assertEqual(tui.contexts, [source, fork])

    def test_headless_invocation_uses_app_provider_tools_and_hooks(self):
        app = ex6.App()
        finished = threading.Event()
        turns = []
        tool_turns = []

        def echo(ctx: ex6.Context, text: str) -> str:
            self.assertIs(ctx.app, app)
            return text

        def provider(ctx):
            if len(turns) == 0:
                yield ex6.LLMResult(tool_calls=[
                    {"id": "call1", "name": "echo", "args": {"text": "hello"}},
                ])
            else:
                yield ex6.ResponseChunk("text", "done")
                yield ex6.LLMResult()

        app.overrides["invoke_llm"] = provider
        app.after_llm_turn(lambda ctx: turns.append(ctx))
        ctx = ex6.Context(app, "headless", "test",
                          messages=[ex6.Message("system", "test", tools=[echo])])

        def after_turn(ctx):
            if not ctx.llm_result.tool_calls:
                ctx.schedule(finished.set)

        ctx.after_llm_turn(after_turn)
        ctx.after_tool_calls(lambda ctx: tool_turns.append(ctx))
        ctx.invoke("go")
        self.assertTrue(finished.wait(5))
        self.assertFalse(ctx.is_running())
        self.assertIsNone(app.tui)
        self.assertEqual(turns, [ctx, ctx])
        self.assertEqual(tool_turns, [ctx])
        self.assertEqual(ex6.tool_result_text(ctx.get_messages()[-2].content), "hello")
        self.assertEqual(ctx.get_messages()[-1].content, "done")


if __name__ == "__main__":
    unittest.main()

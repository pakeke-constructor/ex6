import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ex6
from _ex6 import tools


class PatchFileTests(unittest.TestCase):
    def test_update_bodies(self):
        cases = [
            ("old\n", "\n-old\n+new\n", "new\n"),
            ("old", "@@\n-old\n+new", "new"),
            ("a\nb\nc\n", " a\n-b\n+B\n c", "a\nB\nc\n"),
            ("a\nb\nc\n", "-a\n+A\n@@\n-c\n+C", "A\nb\nC\n"),
            ("a\nb\nc\n", " a\n+x\n b\n+y\n c", "a\nx\nb\ny\nc\n"),
            ("a\nb\n", " a\n+x\n b", "a\nx\nb\n"),
            ("a\nb\n", "-b", "a\n"),
            ("a\n", "-a", ""),
            ("a\n", "+b", "a\nb\n"),
            ("", "+a", "a\n"),
            ("a\na\n", "-a\n+b", "b\na\n"),
            ("a\na\n", "-a\n+b\n*** End of File", "a\nb\n"),
            ("a\nb\n", "+c\n*** End of File", "a\nb\nc\n"),
            ("class A\nx\nclass B\nx\n", "@@ class B\n-x\n+y", "class A\nx\nclass B\ny\n"),
            ("class A\ndef f\nx\n", "@@ class A\n@@ def f\n-x\n+y", "class A\ndef f\ny\n"),
            (" a \nb \n", " a\n-b\n+c", " a \nc\n"),
            ("  a\n  b\n", " a\n-b\n+c", "  a\nc\n"),
            (" old\nold\n", "-old\n+new", " old\nnew\n"),
            ("a\n\nb\n", " a\n\n-b\n+c", "a\n\nc\n"),
            ("a\r\nb\r\n", "-b\r\n+c\r\n", "a\r\nc\r\n"),
            ("a\r\nb", "-b\n+c", "a\r\nc"),
            ("a\rb\r", "-b\n+c", "a\rc\r"),
            ("old\n", '-old\n+print("\\n")', 'print("\\n")\n'),
            ("a\n", " a\n+", "a\n\n"),
        ]
        for old, body, expected in cases:
            with self.subTest(body=body, old=old):
                self.assertEqual(tools._apply_file_patch(old, body), expected)

    def test_invalid_bodies(self):
        bodies = ["", "@@", " a", "...", "*** Begin Patch", "*** Delete File: x",
                  "@@ -1,1 +1,1 @@\n-a\n+b", "-missing\n+b", "-a\n+b\n@@",
                  "-a\n+b\n*** End of File\n+c", "@@ missing\n-a\n+b",
                  "-a\n+b\n*** End of File", "-a\n+b\n@@\n-missing\n+c",
                  "-b\n+c\n@@\n-a\n+d", "-a\n-b\n-c\n+d"]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                tools._apply_file_patch("a\nb\n", body)

    def test_tool_guards_and_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file.py"
            original = b"old\r\n"
            path.write_bytes(original)
            app = ex6.App()
            ctx = app.create_context("test", model="test", cwd=directory)
            with patch.object(tools, "approve", return_value=None) as approve:
                with self.assertRaisesRegex(ValueError, "Must read"):
                    tools.patch_file(ctx, "file.py", "-old\n+new")
                approve.assert_not_called()
                ctx.mark_file_read("file.py", [1])
                with self.assertRaises(ValueError):
                    tools.patch_file(ctx, "file.py", "-old\n+new\n@@\n-missing\n+x")
                approve.assert_not_called()
                self.assertEqual(path.read_bytes(), original)
                approve.return_value = "no"
                with self.assertRaisesRegex(ValueError, "denied"):
                    tools.patch_file(ctx, "file.py", "-old\n+new")
                self.assertEqual(path.read_bytes(), original)
                approve.return_value = None
                self.assertEqual(tools.patch_file(ctx, "file.py", "-old\n+new"), "Patched file.py")
                self.assertEqual(path.read_bytes(), b"new\r\n")
                self.assertTrue(ctx.has_read_file("file.py"))
                self.assertEqual(ctx.get_line_snapshot("file.py"), {})
                self.assertIn("render_extra", approve.call_args.kwargs)
                path.write_bytes(b"changed\n")
                with self.assertRaisesRegex(ValueError, "Must read"):
                    tools.patch_file(ctx, "file.py", "-changed\n+new")
                with self.assertRaisesRegex(ValueError, "Must read"):
                    tools.patch_file(ctx, "missing.py", "+new")
                self.assertFalse((Path(directory) / "missing.py").exists())

    def test_registration(self):
        from _ex6.agents import MAIN_TOOLS
        self.assertIn(tools.patch_file, MAIN_TOOLS)
        self.assertIn(tools.write_file, MAIN_TOOLS)
        self.assertNotIn(tools.edit_file, MAIN_TOOLS)
        schema = ex6.tool_to_schema("patch_file", tools.patch_file)["function"]
        self.assertEqual(schema["parameters"]["required"], ["file", "patch"])


if __name__ == "__main__":
    unittest.main()

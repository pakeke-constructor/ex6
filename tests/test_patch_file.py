from _ex6 import tools


# Simplicity matters: if a test is too complex, don't bother testing it here.
def run():
    content = "class A\ndef f\nold\n"
    patch = "@@ class A\n@@ def f\n-old\n+new\n+x\n*** End of File"
    assert tools._apply_file_patch(content, patch) == "class A\ndef f\nnew\nx\n"
    assert tools._apply_file_patch("old\r\n", "-old\n+new") == "new\r\n"
    try:
        tools._apply_file_patch("old\n", "-missing\n+new")
    except ValueError:
        pass
    else:
        assert False, "Invalid patch should be rejected"


run()

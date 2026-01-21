import ex6
import time


def read_file(ctx: ex6.Context, tool_call_id: str, path: str):
    """Read and return contents of a file at the given path."""
    with open(path, "r") as f:
        content = f.read()
    ctx.add_tool_result(tool_call_id, content)



def ask_user(ctx: ex6.Context, tool_call_id: str, question: str):
    """Ask user a question and wait for their response. Blocks until answered."""
    result = ""

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        nonlocal result
        x, y, w, h = r
        buf.puts(x, y, f"? {question}", txt_color='yellow')
        typed = inpt.consume_text()
        if typed:
            result = (result or "") + typed
        if inpt.consume('KEY_BACKSPACE') and result:
            result = result[:-1]
        if inpt.consume('KEY_ENTER') and result:
            ctx.input_stack.pop()
        buf.puts(x + 2, y + 1, "> " + result + "█", txt_color='white')

    ctx.push_ui(draw)

    while draw in ctx.input_stack:
        time.sleep(0.05)

    ctx.add_tool_result(tool_call_id, result)


tool_system = ex6.Message(
    role="system",
    content="You can read files (read_file) or ask the user questions (ask_user).",
    tools={"read_file": read_file, "ask_user": ask_user}
)

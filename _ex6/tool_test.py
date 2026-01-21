import ex6
import time


def read_file(ctx: ex6.Context, tool_call_id: str, path: str):
    """Read and return contents of a file at the given path."""
    with open(path, "r") as f:
        content = f.read()
    ctx.add_tool_result(tool_call_id, content)



def ask_user(ctx: ex6.Context, tool_call_id: str, question: str):
    """Ask user a question and wait for their response. Blocks until answered."""
    result = [None]

    def on_submit(text):
        result[0] = text
        ctx.input_stack.pop()

    input_draw = ex6.make_input(on_submit)

    def draw(buf: ex6.ScreenBuffer, inpt, r):
        x, y, w, h = r
        buf.puts(x, y, f"? {question}", txt_color='yellow')
        input_draw(buf, inpt, (x + 2, y + 1, w - 2, 1))

    ctx.push_ui(draw)

    while draw in ctx.input_stack:
        time.sleep(0.05)

    assert result[0]
    ctx.add_tool_result(tool_call_id, result[0])


tool_system = ex6.Message(
    role="system",
    content="You can read files (read_file) or ask the user questions (ask_user).",
    tools={"read_file": read_file, "ask_user": ask_user}
)

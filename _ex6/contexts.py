


from _ex6 import provider
from _ex6.tools import read_headers, read_function, glob, grep, search, write_file, edit_file
import ex6
from ex6 import Context, Message
import time
import math




def ask_user(ctx: ex6.Context, question: str) -> str:
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

    return result[0] or ""


coding_agent_system_prompt = ex6.Message(
role ="system",
content="""
# Role and Goal:
You are an intelligent coding assistant, working alongside an experienced engineer.
You will be asked to assist with activities such as debugging code, refactoring functions, and implementing new solutions to the engineer's problemms.

# Guidelines:
- Be as concise as possible.
- Avoid bloated "apologetic language" like "You are absolutely right!"
- If you are using an API or module, you *MUST* look for the actual function/class definition before you use it. Sometimes the user will provide a tool to search the docs.
"""
)



MODEL = "openai/gpt-5.1-codex-mini"


Context("reader", messages=[
    coding_agent_system_prompt,
    provider.code_mode([read_file, glob, grep, search, read_headers, read_function]),
], model=MODEL)



Context("coder", messages=[
    coding_agent_system_prompt,
    provider.code_mode([read_file, glob, grep, search, read_headers, read_function, write_file, edit_file]),
], model=MODEL)



ex6.state.current = c1





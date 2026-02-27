

from _ex6.provider import tool_system_prompt
from _ex6.tools import read_headers, read_function
import ex6
from ex6 import Context, Message
import time
import math



def read_file(ctx: ex6.Context, path: str) -> str:
    """Read and return contents of a file at the given path."""
    time.sleep(3)
    with open(path, "r") as f:
        return f.read()



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



MODEL = "openrouter/openai/gpt-5.1-codex-mini"

c1 = Context("ctx1", messages=[
    coding_agent_system_prompt,
    tool_system_prompt,
    Message(role="system", content="You are helpful."),
    Message(role="user", content="hello"),
    Message(role="assistant", content="Hi! How can I help?"),
], model=MODEL)

Context("ctx2", model=MODEL)
Context("foobar", model=MODEL)


# Example context with file-read tool (code-mode)
Context("file_reader", messages=[
    coding_agent_system_prompt,
    tool_system_prompt,
    Message(role="system", content="", tools={"read_file": read_file}),
], model=MODEL)

# Context with code-reading tools
Context("code_reader", messages=[
    coding_agent_system_prompt,
    tool_system_prompt,
    Message(role="system", content="", tools={
        "read_headers": read_headers,
        "read_function": read_function,
    }),
], model=MODEL)

ex6.state.current = c1




s = '''

SPINNER
SPINNER
SPINNER

# hello.
*I am italic.*
### i am a 3rd heading!
and im a `func()` call.
- a
- bbb
- cccd


```python
def func(x: int):
    for i in range(10):
        print(i)
        break
    return 0.0
```


```tools
read_file("test.txt")
```



'''

#@ex6.override
def invoke_llm(ctx):
    """Override this to use real LLM."""
    time.sleep(2)
    for i in range(40):
        yield ex6.ResponseChunk("text", "tok ")
        time.sleep(0.03)
    yield ex6.ResponseChunk("text", s)



def render_spinner(buf: ex6.ScreenBuffer, x: int, y: int, w: int) -> int:
    txt = "spinner! " + ("\\|/—"[math.floor(time.time()*5) % 4])
    buf.puts(x, y, txt, txt_color='red')
    lines_used = 1
    return lines_used


@ex6.output_renderer
def example_renderer(role: str, output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    # Replace lines containing "SPINNER" with a red spinner
    for i, line in enumerate(output):
        if isinstance(line, str) and "SPINNER" in line:
            output[i] = render_spinner



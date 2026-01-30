

from _ex6.litellm import tool_system_prompt
import ex6
from ex6 import Context, Message
import time
import math



def read_file(ctx: ex6.Context, path: str) -> str:
    """Read and return contents of a file at the given path."""
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


tool_system = ex6.Message(
    role="system",
    content="You can read files (read_file) or ask the user questions (ask_user).",
    tools={"read_file": read_file, "ask_user": ask_user}
)



MODEL = "openrouter/openai/gpt-5.1-codex-mini"

c1 = Context("ctx1", messages=[
    tool_system_prompt,
    Message(role="system", content="You are helpful."),
    Message(role="user", content="hello"),
    Message(role="assistant", content="Hi! How can I help?"),
], model=MODEL)

Context("ctx2", model=MODEL)
Context("foobar", model=MODEL)


# Example context with file-read tool (code-mode)
Context("file_reader", messages=[
    tool_system_prompt,
    Message(role="system", content="You can read files.", tools={"read_file": read_file}),
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
def example_renderer(output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    # Replace lines containing "SPINNER" with a red spinner
    for i, line in enumerate(output):
        if isinstance(line, str) and "SPINNER" in line:
            # if line contains `SPINNER`, replace line with a spinner!
            output[i] = render_spinner



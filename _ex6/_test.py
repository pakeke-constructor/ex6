

from _ex6 import provider
from _ex6.code_mode import make_code_mode_system_prompt
from _ex6.tools import read_headers, read_function, glob, search, write_file, edit_file, read_file
import ex6
from ex6 import Context, Message
import time
import math




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

c1 = Context("ctx1", messages=[
    coding_agent_system_prompt,
    Message(role="system", content="You are helpful."),
    Message(role="user", content="hello"),
    Message(role="assistant", content="Hi! How can I help?"),
], model=MODEL)

Context("ctx2", model=MODEL)
Context("foobar", model=MODEL)


Context("reader", messages=[
    coding_agent_system_prompt,
    make_code_mode_system_prompt([read_file, glob, search, read_headers, read_function]),
], model=MODEL)



Context("ctx_1", messages=[
    coding_agent_system_prompt,
    make_code_mode_system_prompt([glob, search, read_headers, read_function]),
], model=MODEL)




Context("coder", messages=[
    coding_agent_system_prompt,
    make_code_mode_system_prompt([read_file, glob, search, read_headers, read_function, write_file, edit_file]),
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
def example_renderer(msg: ex6.Message, output: list[ex6.OutputLine], ctx: ex6.Context) -> None:
    # Replace lines containing "SPINNER" with a red spinner
    for i, line in enumerate(output):
        if isinstance(line, str) and "SPINNER" in line:
            output[i] = render_spinner



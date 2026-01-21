
import ex6
from ex6 import Context, Message

from _ex6.tool_test import read_file


MODEL = "z-ai/glm-4.7-flash"

c1 = Context("ctx1", messages=[
    Message(role="system", content="You are helpful."),
    Message(role="user", content="hello"),
    Message(role="assistant", content="Hi! How can I help?"),
], model=MODEL)

Context("ctx2", model=MODEL)
Context("foobar", model=MODEL)

# Example context with file-read tool
Context("file_reader", messages=[
    Message(role="system", content="You can read files.", tools={"read_file": read_file}),
    Message(role="user", content="Read .ex6/test_ctxs.py"),
], model=MODEL)


ex6.state.current = c1



import ex6
from ex6 import Context, Message

from _ex6.tool_test import read_file


c1 = Context("ctx1", messages=[
    Message(role="system", content="You are helpful."),
    Message(role="user", content="hello"),
    Message(role="assistant", content="Hi! How can I help?"),
])
Context("ctx2", model="sonnet-4")
Context("foobar", cost=0.08)

# Example context with file-read tool
Context("file_reader", messages=[
    Message(role="system", content="You can read files.", tools={"read_file": read_file}),
    Message(role="user", content="Read .ex6/test_ctxs.py"),
])

ex6.state.current = c1


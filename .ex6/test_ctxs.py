
import ex6
from ex6 import Context, Message




c1 = Context("ctx1", messages=[
    Message(role="system", content="You are helpful."),
    Message(role="user", content="hello"),
    Message(role="assistant", content="Hi! How can I help?"),
])
Context("ctx2", model="sonnet-4")
Context("foobar", cost=0.08)


ex6.state.current = c1




'''

Checkpoint tools.

PURPOSE:
Essentially give agents a way to "clean" their context windows up to a certain checkpoint.
eg:

user: Implement auth for StudentService

assistant: Sure:
tools: checkpoint(
    objective = "exploring codebase, finding auth for studentService",
)

tools: read_file(...)
tools: search(...)
tools: read_file(...)
tools: read_file(...)

assistant: A good solution is ..... foobar ...

tools: condense(
    findings = "AuthService does xyz, ... You can't do foobar because qux, `barfoo` is the best solution.",
    tools = r"""
    # stuff that the condenser wants to inject into the condensed output:
    read_file("a.py")
    read_headers("b.py")
    read_body("c.py", "get_user_id")
    say("a.py")
    """
)



ADVANTAGE OVER SUBAGENTS/FORKING:
Agents don't know how "big" a task is until they actually start working on it.
Forking is pre-emptive. "I predict this task is big; so i fork."
Checkpoints are lazy: "If this task gets too big, I'll collapse back to checkpoint"


'''


def checkpoint(objective: str):
    pass


def condense(objective: str):
    pass


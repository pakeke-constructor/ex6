
from typing import Optional
from src import ex6


@ex6.command
def clear(name: Optional[str]):
    pass
    # clears context
    # removes all messages except for the STARTING system-prompt messages.
    # ie loop over the beginning of ctx.messages;
    # when you get to non sys prompt; prune everything.

    # (if name is None, uses current-context.)


@ex6.command
def delete(name: Optional[str]):
    pass
    # deletes context
    # (if name is None, uses current-context.)


@ex6.command
def fork(name: Optional[str]):
    pass
    # forks context
    # (if name is None, uses current-context.)






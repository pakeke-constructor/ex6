import ex6

def read_file(ctx: ex6.Context, tool_call_id: str, path: str):
    """Read and return contents of a file at the given path."""
    with open(path, "r") as f:
        content = f.read()
    ctx.add_tool_result(tool_call_id, content)
    ctx.request_continue()

tool_system = ex6.Message(
    role="system",
    content="You can read files using the read_file tool. Call it with a path argument.",
    tools={"read_file": read_file}
)

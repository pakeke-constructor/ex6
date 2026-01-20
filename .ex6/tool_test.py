import ex6

def read_file(ctx: ex6.Context, path: str) -> str:
    with open(path, "r") as f:
        return f.read()

tool_system = ex6.Message(
    role="system",
    content="You can read files using the read_file tool. Call it with a path argument.",
    tools={"read_file": read_file}
)

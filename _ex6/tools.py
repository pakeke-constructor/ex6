


'''

Basic tools for ex6.
includes:

- reading/writing/updating files
- reading/writing function bodies
- reading class/func headers
- glob
- pulling skills


'''

import ex6
import os
import ast
import glob as _glob


def _node_start(node):
    """Line index where a node starts (includes decorators)."""
    if hasattr(node, 'decorator_list') and node.decorator_list:
        return node.decorator_list[0].lineno - 1
    return node.lineno - 1


def read_file(ctx: ex6.Context, file: str) -> str:
    "Reads a file and returns its contents."
    with open(file, "r") as f:
        return f.read()


def write_file(ctx: ex6.Context, file: str, content: str) -> str:
    """
    Writes content to a file.
    If the file exists, it is cleared.
    If the file doesn't exist, a new file is created.
    """
    d = os.path.dirname(file)
    if d: os.makedirs(d, exist_ok=True)
    with open(file, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {file}"


def update_file(ctx: ex6.Context, file: str, search: str, replace: str) -> str:
    """Updates a file by searching and replacing a string."""
    with open(file, "r") as f:
        content = f.read()
    if search not in content:
        return f"ERROR: search string not found in {file}"
    content = content.replace(search, replace, 1)
    with open(file, "w") as f:
        f.write(content)
    return f"Updated {file}"


def find_files(ctx: ex6.Context, pattern: str) -> str:
    """Glob for files matching a pattern. Returns newline-separated paths."""
    matches = _glob.glob(pattern, recursive=True)
    return "\n".join(matches) if matches else "No matches."


def read_headers(ctx: ex6.Context, file: str) -> str:
    """Read class/function signatures from a file (no bodies)."""
    with open(file, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    lines = source.split('\n')
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sig = lines[node.lineno - 1].rstrip()
            doc = ast.get_docstring(node)
            if doc:
                sig += f'  # {doc.split(chr(10))[0]}'
            out.append(sig)
    return "\n".join(out) if out else "No classes/functions found."


def read_function(ctx: ex6.Context, file: str, name: str) -> str:
    """Read a function or class body by name from a file."""
    with open(file, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    lines = source.split('\n')
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return "\n".join(lines[_node_start(node) : node.end_lineno])
    return f"ERROR: '{name}' not found in {file}"


def write_function(ctx: ex6.Context, file: str, name: str, code: str) -> str:
    """Replace a function or class by name in a file with new code."""
    with open(file, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    lines = source.split('\n')
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            before = lines[:_node_start(node)]
            after = lines[node.end_lineno:]
            with open(file, "w") as f:
                f.write("\n".join(before + code.split('\n') + after))
            return f"Replaced '{name}' in {file}"
    return f"ERROR: '{name}' not found in {file}"




'''

Basic tools for ex6.
includes:

- reading/writing/updating files
- reading function bodies
- reading class/func headers
- glob


'''

import ex6
import os
import re
import glob as _glob
import importlib
import tree_sitter


LANG_MODULES = {
    '.py': 'tree_sitter_python', '.pyw': 'tree_sitter_python',
    '.js': 'tree_sitter_javascript', '.mjs': 'tree_sitter_javascript',
    '.ts': 'tree_sitter_typescript', '.tsx': 'tree_sitter_typescript',
    '.jsx': 'tree_sitter_javascript',
    '.go': 'tree_sitter_go',
    '.rs': 'tree_sitter_rust',
    '.c': 'tree_sitter_c', '.h': 'tree_sitter_c',
    '.cpp': 'tree_sitter_cpp', '.hpp': 'tree_sitter_cpp', '.cc': 'tree_sitter_cpp',
    '.java': 'tree_sitter_java',
    '.rb': 'tree_sitter_ruby',
    '.cs': 'tree_sitter_c_sharp',
    '.lua': 'tree_sitter_lua',
    '.kt': 'tree_sitter_kotlin', '.kts': 'tree_sitter_kotlin',
}

DEFINITION_TYPES = {
    'tree_sitter_python': ['function_definition', 'class_definition'],
    'tree_sitter_javascript': ['function_declaration', 'class_declaration', 'method_definition'],
    'tree_sitter_typescript': ['function_declaration', 'class_declaration', 'method_definition'],
    'tree_sitter_go': ['function_declaration', 'method_declaration', 'type_declaration'],
    'tree_sitter_rust': ['function_item', 'struct_item', 'enum_item', 'impl_item', 'trait_item'],
    'tree_sitter_c': ['function_definition', 'struct_specifier'],
    'tree_sitter_cpp': ['function_definition', 'class_specifier', 'struct_specifier'],
    'tree_sitter_java': ['method_declaration', 'class_declaration', 'interface_declaration'],
    'tree_sitter_ruby': ['method', 'class', 'module'],
    'tree_sitter_c_sharp': ['method_declaration', 'class_declaration', 'interface_declaration'],
    'tree_sitter_lua': ['function_declaration', 'variable_declaration', 'assignment_statement'],
    'tree_sitter_kotlin': ['function_declaration', 'class_declaration', 'object_declaration', 'companion_object'],
}


def _parse_file(file):
    ext = os.path.splitext(file)[1].lower()
    mod_name = LANG_MODULES.get(ext)
    if not mod_name:
        raise ValueError(f"Unsupported file type: {ext}")
    mod = importlib.import_module(mod_name)
    lang = tree_sitter.Language(mod.language())
    parser = tree_sitter.Parser(lang)
    with open(file, "rb") as f:
        source = f.read()
    return parser.parse(source), source, mod_name


def _get_name(node):
    """Get the name of a definition node."""
    n = node.child_by_field_name('name')
    return n.text.decode() if n else None


def _signature_generic(node, source):
    body = node.child_by_field_name('body')
    if body:
        return source[node.start_byte:body.start_byte].decode().rstrip().rstrip(':')
    return source[node.start_byte:node.end_byte].decode().split('\n')[0]


def _signature_python(node, source):
    sig = _signature_generic(node, source)
    body = node.child_by_field_name('body')
    if body and body.children:
        first = body.children[0]
        if first.type in ('expression_statement', 'comment'):
            child = first.children[0] if first.children else first
            if child.type in ('string', 'comment'):
                doc = child.text.decode().strip().strip('"\' \n')
                first_line = doc.split('\n')[0].strip()
                if first_line:
                    sig += f'  # {first_line}'
    return sig


def _signature_lua(node, source):
    sig = source[node.start_byte:node.end_byte].decode().split('\n')[0]
    annotations = []
    sib = node.prev_sibling
    while sib and sib.type == 'comment' and sib.text.decode().startswith('---'):
        annotations.append(sib.text.decode())
        sib = sib.prev_sibling
    if annotations:
        annotations.reverse()
        return '\n'.join(annotations) + '\n' + sig
    return sig


def _signature_kotlin(node, source):
    for child in node.children:
        if child.type in ('function_body', 'class_body', 'enum_class_body'):
            return source[node.start_byte:child.start_byte].decode().rstrip()
    return source[node.start_byte:node.end_byte].decode().split('\n')[0]


_SIGNATURE_FNS = {
    'tree_sitter_python': _signature_python,
    'tree_sitter_lua': _signature_lua,
    'tree_sitter_kotlin': _signature_kotlin,
}


def _signature(node, source, mod_name):
    return _SIGNATURE_FNS.get(mod_name, _signature_generic)(node, source)


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


def edit_file(ctx: ex6.Context, file: str, search: str, replace: str) -> str:
    """Edit a file by searching and replacing a string."""
    with open(file, "r") as f:
        content = f.read()
    if search not in content:
        return f"ERROR: search string not found in {file}"
    content = content.replace(search, replace, 1)
    with open(file, "w") as f:
        f.write(content)
    return f"Updated {file}"


def glob(ctx: ex6.Context, pattern: str) -> str:
    """Find files matching a glob pattern (recursive). Returns newline-separated paths."""
    matches = _glob.glob(pattern, recursive=True)
    return "\n".join(matches) if matches else "No matches."


def grep(ctx: ex6.Context, pattern: str, path: str = ".") -> str:
    """Search file contents for a regex pattern. Returns matching lines with file:line: prefix."""
    results = []
    path = os.path.abspath(path)
    if os.path.isfile(path):
        files = [path]
    else:
        files = [os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs]
    regex = re.compile(pattern)
    for f in files:
        try:
            with open(f, "r", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if regex.search(line):
                        results.append(f"{f}:{i}: {line.rstrip()}")
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(results) if results else "No matches."


def _read_headers_lua(tree, source):
    def_types = DEFINITION_TYPES['tree_sitter_lua']
    out = []

    def collect(node):
        for child in node.children:
            if child.type in def_types:
                if out:
                    out.append("")
                out.append(_signature_lua(child, source))
                if child.type not in ('variable_declaration', 'assignment_statement'):
                    collect(child)
            else:
                collect(child)

    collect(tree.root_node)
    return "\n".join(out) if out else "No classes/functions found."


_SKIP_DIRS = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.tox', '.mypy_cache', '.pytest_cache', 'dist', 'build', '.egg-info'}


def search(ctx: ex6.Context, pattern: str, match: str = "**/*", max_results: int = 10) -> str:
    """Search file contents for a regex pattern, filtered by glob. Returns matching lines with context."""
    regex = re.compile(pattern)
    matched_files = _glob.glob(match, recursive=True)
    results = []
    for f in matched_files:
        if not os.path.isfile(f):
            continue
        parts = f.replace("\\", "/").split("/")
        if any(p in _SKIP_DIRS for p in parts):
            continue
        try:
            with open(f, "r", errors="ignore") as fh:
                for i, line in enumerate(fh, 1):
                    if regex.search(line):
                        results.append(f"{f}:{i}: {line.rstrip()}")
                        if len(results) >= max_results:
                            return "\n".join(results) + f"\n... (capped at {max_results} results)"
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(results) if results else "No matches."


def read_headers(ctx: ex6.Context, file: str) -> str:
    """Read class/function signatures from a file (no bodies)."""
    tree, source, mod_name = _parse_file(file)
    if mod_name == 'tree_sitter_lua':
        return _read_headers_lua(tree, source)
    def_types = DEFINITION_TYPES.get(mod_name, [])
    out = []

    def collect(node, indent=0):
        prefix = "  " * indent
        for child in node.children:
            if child.type in def_types:
                if indent == 0 and out:
                    out.append("")  # gap between top-level defs
                out.append(prefix + _signature(child, source, mod_name).strip())
                collect(child, indent + 1)
            else:
                collect(child, indent)

    collect(tree.root_node)
    return "\n".join(out) if out else "No classes/functions found."


def read_function(ctx: ex6.Context, file: str, name: str) -> str:
    """Read a function or class body by name from a file."""
    tree, source, mod_name = _parse_file(file)
    def_types = DEFINITION_TYPES.get(mod_name, [])

    def find(node):
        for child in node.children:
            if child.type in def_types and _get_name(child) == name:
                return source[child.start_byte:child.end_byte].decode()
            result = find(child)
            if result:
                return result
        return None

    result = find(tree.root_node)
    return result if result else f"ERROR: '{name}' not found in {file}"

